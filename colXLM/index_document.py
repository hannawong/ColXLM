from colXLM.modeling.colbert import ColBERT,ColXLM
from colXLM.modeling.inference import ModelInference
from colXLM.utils.utils import load_checkpoint
import pickle as pkl
import ujson, os
import torch

def get_embedding(document,inference):
    embs = inference.docFromText([document])[0]
    return embs

colbert = ColBERT.from_pretrained('bert-base-multilingual-uncased',
                                      query_maxlen=32,
                                      doc_maxlen=180,
                                      dim=128,
                                      similarity_metric="l2",
                                      mask_punctuation=True)

colbert = colbert.to("cuda")
print("#> Loading model checkpoint.")
checkpoint = load_checkpoint("/data/jiayu_xiao/project/wzh/ColXLM/MSMARCO-psg/train.py/msmarco.psg.l2/checkpoints/colbert.dnn", colbert, do_print=True)
colbert.eval()

inference = ModelInference(colbert, amp=-1)

Doc = open("./colXLM/Dataset/documents.tsv")
doclens = []
doc_emb_concat = None

for i in range(11979):
    print(i)
    doc = Doc.readline().split("\t")
    pid = doc[0]
    document = doc[1]
    doc_emb = get_embedding(document, inference)
    doclens.append(doc_emb.shape[0])
    if doc_emb_concat == None:
        doc_emb_concat = doc_emb
    else:
        doc_emb_concat = torch.cat([doc_emb_concat, doc_emb],axis = 0)

output_path = os.path.join("/data/jiayu_xiao/project/wzh/ColXLM/colXLM/indexes", "0.pt")
doclens_path = os.path.join("/data/jiayu_xiao/project/wzh/ColXLM/colXLM/indexes", 'doclens.{}.json'.format(0))
torch.save(doc_emb_concat, output_path)
with open(doclens_path, 'w') as output_doclens:
            ujson.dump(doclens, output_doclens)
