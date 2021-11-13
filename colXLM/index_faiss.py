import faiss
import os
import torch

DIM = 128
index = faiss.IndexFlatL2(128)
doc_emb = torch.load("/data/jiayu_xiao/project/wzh/ColXLM/colXLM/indexes/0.pt").cpu().numpy()
index.add(doc_emb)
output_path = os.path.join("/data/jiayu_xiao/project/wzh/ColXLM/colXLM/indexes", "faiss_l2")
faiss.write_index(index, output_path)
