from colXLM.modeling.colbert import ColBERT
from colXLM.modeling.tokenization import QueryTokenizer, DocTokenizer
from colXLM.utils.amp import MixedPrecisionManager
from colXLM.parameters import DEVICE
from colXLM.modeling.inference import ModelInference
from colXLM.utils.utils import load_checkpoint
import pickle as pkl


colbert = ColBERT.from_pretrained('bert-base-uncased',
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

def get_embedding(document):
    embs = inference.docFromText([document])[0]
    return embs

Doc = open("./colXLM/Dataset/documents.tsv")
doc_emb = {}
for i in range(11979):
    print(i)
    doc = Doc.readline().split("\t")
    pid = doc[0]
    document = doc[1]
    doc_emb[pid] = get_embedding(document)

with open("./colXLM/Dataset/doc_emb.pkl","wb") as file:
    pkl.dump(doc_emb,file)
