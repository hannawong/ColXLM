import faiss
import os
import torch

from colXLM.utils.parser import Arguments

parser = Arguments(description='index document into faiss')
parser.add_argument('--dim', dest='dim', default=128)
parser.add_argument('--index_path',dest = 'index_path',default = '/data/jiayu_xiao/project/wzh/ColXLM/colXLM/indexes')
parser.add_argument('--faiss_name',dest = "faiss_name",default = "faiss_l2")
args = parser.parse()

doc_emb = torch.load(os.path.join(args.index_path,"0.pt")).cpu().numpy()
index = faiss.IndexFlatL2(int(args.dim))
index.add(doc_emb)
output_path = os.path.join(args.index_path, args.faiss_name)
faiss.write_index(index, output_path)
