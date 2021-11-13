import pickle as pkl
import numpy as np
from colXLM.modeling.colbert import ColBERT
from colXLM.modeling.tokenization import QueryTokenizer, DocTokenizer
from colXLM.utils.amp import MixedPrecisionManager
from colXLM.parameters import DEVICE
from colXLM.modeling.inference import ModelInference
from colXLM.utils.utils import load_checkpoint
import pickle as pkl
'''
collection = open("/data/jiayu_xiao/project/wzh/ColXLM/colXLM/Dataset/documents.tsv")
for i in range(11979):
    co = collection.readline().split("\t")
    if(int(co[0])==1182193):
        print(co[1])
'''

def calc_similarity(query_emb, doc_emb):  ##query_emb: [32,128], doc_emb:[56,128]
    query_token_num = query_emb.shape[0]
    score = 0.0
    for i in range(query_token_num): ## 遍历整个query tokens
        max_sim = 0.0
        for j in range(doc_emb.shape[0]): ##遍历所有的doc tokens
            sim = np.matmul(query_emb[i].cpu().T,doc_emb[j].cpu())
            if sim.item() > max_sim:
                max_sim = sim
        score += max_sim
    return score


query = "Définition du récepteur aux androgènes"
colbert = ColBERT.from_pretrained('xlm-mlm-tlm-xnli15-1024',
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

def get_embedding(query):
    embs = inference.queryFromText([query])[0]
    return embs


with open("./colXLM/Dataset/doc_emb.pkl","rb") as file:
    doc_embeddings = pkl.load(file)

query_emb = get_embedding(query)

doc_scores = {}
cnt = 0
length = len(doc_embeddings.keys())
for doc_key in list(doc_embeddings.keys()):
    cnt += 1
    print(cnt/length)
    score = calc_similarity(query_emb, doc_embeddings[doc_key])
    doc_scores[doc_key] = score

print(sorted(doc_scores.items(), key = lambda kv:(kv[1], kv[0]),reverse=True))   
    

