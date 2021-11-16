import faiss
from itertools import accumulate
import ujson
import os
import torch
import time

from colXLM.modeling.colbert import ColBERT
from colXLM.parameters import DEVICE
from colXLM.modeling.inference import ModelInference
from colXLM.utils.utils import load_checkpoint
from colXLM.utils.parser import Arguments

parser = Arguments(description='retrieve top k relevant documents')
parser.add_argument('--checkpoint_path', dest='checkpoint_path', default='/data/jiayu_xiao/project/wzh/ColXLM/MSMARCO-psg/train.py/msmarco.psg.l2/checkpoints/colbert.dnn')
parser.add_argument('--index_path',dest = 'index_path',default = '/data/jiayu_xiao/project/wzh/ColXLM/colXLM/indexes')
parser.add_argument('--faiss_name',dest = "faiss_name",default = "faiss_l2")
parser.add_argument('--k',dest = 'k',default = 20)
args = parser.parse()

k = int(args.k)
BSIZE = 1 << 14

def doc_index(doclens):
    dic = {}
    cnt = 0
    for i,doclen in enumerate(doclens):
        for j in range(doclen):
            dic[cnt] = i
            cnt += 1
    return dic

def get_embedding(query,inference):
    embs = inference.queryFromText([query])[0]
    return embs

class IndexRanker():
    def __init__(self, tensor, doclens):
        self.tensor = tensor
        self.doclens = doclens

        self.maxsim_dtype = torch.float32
        self.doclens_pfxsum = [0] + list(accumulate(self.doclens))

        self.doclens = torch.tensor(self.doclens)
        self.doclens_pfxsum = torch.tensor(self.doclens_pfxsum)

        self.dim = self.tensor.size(-1)

        self.strides = [torch_percentile(self.doclens, p) for p in [90]]
        self.strides.append(self.doclens.max().item())
        self.strides = sorted(list(set(self.strides)))

        print(f"#> Using strides {self.strides}..")

        self.views = self._create_views(self.tensor)
        self.buffers = self._create_buffers(BSIZE, self.tensor.dtype, {'cpu', 'cuda:0'})

    def _create_views(self, tensor):
        views = []

        for stride in self.strides:
            outdim = tensor.size(0) - stride + 1
            view = torch.as_strided(tensor, (outdim, stride, self.dim), (self.dim, self.dim, 1))
            views.append(view)

        return views

    def _create_buffers(self, max_bsize, dtype, devices):
        buffers = {}

        for device in devices:
            buffers[device] = [torch.zeros(max_bsize, stride, self.dim, dtype=dtype,
                                           device=device, pin_memory=(device == 'cpu'))
                               for stride in self.strides]

        return buffers

    def rank(self, Q, pids, views=None, shift=0):
        assert len(pids) > 0
        assert Q.size(0) in [1, len(pids)]

        Q = Q.contiguous().to(DEVICE).to(dtype=self.maxsim_dtype)

        views = self.views if views is None else views
        VIEWS_DEVICE = views[0].device

        D_buffers = self.buffers[str(VIEWS_DEVICE)]

        raw_pids = pids if type(pids) is list else pids.tolist()
        pids = torch.tensor(pids) if type(pids) is list else pids

        doclens, offsets = self.doclens[pids], self.doclens_pfxsum[pids]

        assignments = (doclens.unsqueeze(1) > torch.tensor(self.strides).unsqueeze(0) + 1e-6).sum(-1)

        one_to_n = torch.arange(len(raw_pids))
        output_pids, output_scores, output_permutation = [], [], []

        for group_idx, stride in enumerate(self.strides):
            locator = (assignments == group_idx)

            if locator.sum() < 1e-5:
                continue

            group_pids, group_doclens, group_offsets = pids[locator], doclens[locator], offsets[locator]
            group_Q = Q if Q.size(0) == 1 else Q[locator]

            group_offsets = group_offsets.to(VIEWS_DEVICE) - shift
            group_offsets_uniq, group_offsets_expand = torch.unique_consecutive(group_offsets, return_inverse=True)

            D_size = group_offsets_uniq.size(0)
            D = torch.index_select(views[group_idx], 0, group_offsets_uniq, out=D_buffers[group_idx][:D_size])
            D = D.to(DEVICE)
            D = D[group_offsets_expand.to(DEVICE)].to(dtype=self.maxsim_dtype)

            mask = torch.arange(stride, device=DEVICE) + 1
            mask = mask.unsqueeze(0) <= group_doclens.to(DEVICE).unsqueeze(-1)

            scores = (D @ group_Q) * mask.unsqueeze(-1)
            scores = scores.max(1).values.sum(-1).cpu()

            output_pids.append(group_pids)
            output_scores.append(scores)
            output_permutation.append(one_to_n[locator])

        output_permutation = torch.cat(output_permutation).sort().indices
        output_pids = torch.cat(output_pids)[output_permutation].tolist()
        output_scores = torch.cat(output_scores)[output_permutation].tolist()

        assert len(raw_pids) == len(output_pids)
        assert len(raw_pids) == len(output_scores)
        assert raw_pids == output_pids

        return output_scores

    def batch_rank(self, all_query_embeddings, all_query_indexes, all_pids, sorted_pids):
        assert sorted_pids is True

        ######

        scores = []
        range_start, range_end = 0, 0

        for pid_offset in range(0, len(self.doclens), 50_000):
            pid_endpos = min(pid_offset + 50_000, len(self.doclens))

            range_start = range_start + (all_pids[range_start:] < pid_offset).sum()
            range_end = range_end + (all_pids[range_end:] < pid_endpos).sum()

            pids = all_pids[range_start:range_end]
            query_indexes = all_query_indexes[range_start:range_end]

            print(f"###--> Got {len(pids)} query--passage pairs in this sub-range {(pid_offset, pid_endpos)}.")

            if len(pids) == 0:
                continue

            print(f"###--> Ranking in batches the pairs #{range_start} through #{range_end} in this sub-range.")

            tensor_offset = self.doclens_pfxsum[pid_offset].item()
            tensor_endpos = self.doclens_pfxsum[pid_endpos].item() + 512

            collection = self.tensor[tensor_offset:tensor_endpos].to(DEVICE)
            views = self._create_views(collection)

            print(f"#> Ranking in batches of {BSIZE} query--passage pairs...")

            for batch_idx, offset in enumerate(range(0, len(pids), BSIZE)):
                if batch_idx % 100 == 0:
                    print("#> Processing batch #{}..".format(batch_idx))

                endpos = offset + BSIZE
                batch_query_index, batch_pids = query_indexes[offset:endpos], pids[offset:endpos]

                Q = all_query_embeddings[batch_query_index]

                scores.extend(self.rank(Q, batch_pids, views, shift=tensor_offset))

        return scores


def torch_percentile(tensor, p):
    assert p in range(1, 100+1)
    assert tensor.dim() == 1

    return tensor.kthvalue(int(p * tensor.size(0) / 100.0)).values.item()


def get_topk(query,inference,index,doc2index,doc_tensor,doclens,query_len): ## TODO: support batch search
    start = time.time()
    query_emb = get_embedding(query,inference).cpu().numpy()[:query_len]
    freq_dic = {}
    D, I = index.search(query_emb, k//2)         

    document_set = set()
    for i in range(I.shape[0]):
        for j in range(I.shape[1]):
            document_set.add(doc2index[I[i][j]])
            if doc2index[I[i][j]] not in freq_dic.keys():
                freq_dic[doc2index[I[i][j]]] = 1
            else:
                freq_dic[doc2index[I[i][j]]] += 1
    print(freq_dic)
    doc_tensor_ = torch.zeros(doc_tensor.shape[0] + 512, doc_tensor.shape[1], dtype=torch.float16)
    doc_tensor_[:doc_tensor.shape[0]] = doc_tensor
    indexranker = IndexRanker(doc_tensor,doclens)
    query_emb = torch.tensor(query_emb)
    query_emb = query_emb.unsqueeze(0)
    query_emb = query_emb.permute(0,2,1)

    score = indexranker.rank(query_emb.cuda(),list(document_set))
    score_sorter = torch.tensor(score).sort(descending=True)
    pids = torch.tensor(list(document_set))[score_sorter.indices].tolist()
    end = time.time()
    print("time elapse during retrieval:",end-start,"s")
    return pids[:k]

def main():
    doclens = ujson.load(open(os.path.join(args.index_path,'doclens.0.json')))
    doc_tensor = torch.load(os.path.join(args.index_path,"0.pt"))
    index = faiss.read_index("/data/jiayu_xiao/project/wzh/ColXLM/colXLM/indexes/faiss_l2")

    doc2index = doc_index(doclens)
    colbert = ColBERT.from_pretrained('bert-base-multilingual-uncased',
                                        query_maxlen=32,
                                        doc_maxlen=180,
                                        dim=128,
                                        similarity_metric="l2",
                                        mask_punctuation=True)
    colbert = colbert.to("cuda")
    print("#> Loading model checkpoint.")
    load_checkpoint(args.checkpoint_path, colbert, do_print=True)
    colbert.eval()
    inference = ModelInference(colbert, amp=-1)
    query = "treating tension headaches without medication"
    tokens = inference.query_tokenizer.tok.tokenize(query)
    print(tokens)
    pids = get_topk(query,inference,index,doc2index,doc_tensor,doclens,len(tokens))
    print(pids)


main()
    