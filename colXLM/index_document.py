import ujson
import os
import torch

from colXLM.modeling.colbert import ColBERT,ColXLM
from colXLM.modeling.inference import ModelInference
from colXLM.utils.utils import load_checkpoint
from colXLM.utils.parser import Arguments

parser = Arguments(description='index document')
parser.add_argument('--checkpoint_path', dest='checkpoint_path', default='/data/jiayu_xiao/project/wzh/ColXLM/MSMARCO-psg/train.py/msmarco.psg.l2/checkpoints/colbert.dnn')
parser.add_argument('--index_path',dest = 'index_path',default = '/data/jiayu_xiao/project/wzh/ColXLM/colXLM/indexes')
parser.add_argument('--doc_path',dest = "doc_path",default = "./colXLM/Dataset/documents.tsv")
args = parser.parse()

mode = "BERT"
def get_embedding(document,inference):
    embs = inference.docFromText([document])[0]
    return embs


def main():
    if mode[:4] == "BERT":
        colbert = ColBERT.from_pretrained('bert-base-multilingual-uncased',
                                            query_maxlen=32,
                                            doc_maxlen=180,
                                            dim=128,
                                            similarity_metric="l2",
                                            mask_punctuation=True)
    if mode == "XLM":
        colbert = ColXLM.from_pretrained('xlm-mlm-tlm-xnli15-1024',
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

    Doc = open(args.doc_path)
    doclens = []
    doc_emb_concat = None

    cnt = 0
    while(True):
        try:
            if cnt % 1000 == 0:
                print(cnt)
            doc = Doc.readline().split("\t")
            document = doc[1]
            doc_emb = get_embedding(document, inference)
            doclens.append(doc_emb.shape[0])
            if doc_emb_concat == None:
                doc_emb_concat = doc_emb
            else:
                doc_emb_concat = torch.cat([doc_emb_concat, doc_emb],axis = 0)
            cnt += 1
        except:
            break

    output_path = os.path.join(args.index_path, "0.pt")
    doclens_path = os.path.join(args.index_path, 'doclens.{}.json'.format(0))
    torch.save(doc_emb_concat, output_path)
    with open(doclens_path, 'w') as output_doclens:
        ujson.dump(doclens, output_doclens)

main()