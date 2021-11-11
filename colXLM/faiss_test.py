import numpy as np
d = 64                           # dimension
nb = 100000                      # database size
nq = 10000                       # nb of queries
np.random.seed(1234)             # make reproducible
xb = np.random.random((nb, d)).astype('float32')
xb[:, 0] += np.arange(nb) / 1000. # # 每一项增加了一个等差数列的对应项数
xq = np.random.random((nq, d)).astype('float32')
xq[:, 0] += np.arange(nq) / 1000.


import faiss                   # make faiss available
index = faiss.IndexFlatL2(64)   # build the index
index.add(xb)                  # add vectors to the index
print(index.ntotal)            # 索引中向量的数量。


k = 4                          # we want to see 4 nearest neighbors
D, I = index.search(xb[:5], k) # sanity check
print(I)
print(D)

print(xq.shape)
D, I = index.search(xq, k)     # actual search
print(D.shape)
print(I.shape)
exit()
print(I[:5])                   # neighbors of the 5 first queries
print(I[-5:])                  # neighbors of the 5 last qu