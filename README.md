# Cross-lingual Information Retrieval Model for Document Search



## Pretraining Phase

We continue pretraining our retrieval-oriented language models from the public XLM checkpoint. Therefore, our cross-lingual LM is implicitly pretrained with three objectives(MLM, TLM, RR)

```sh
sh train.sh
```

## Indexing Phase
In this step, we use the model trained in the pretraining phase to embed every document, and then store the embedding on disk. 

```sh
sh index_document.sh
```

We typically recommend that you use ColXLM for **end-to-end** retrieval, where it directly finds its top-k passages from the full collection. For this, you need FAISS indexing.

#### FAISS Indexing for end-to-end retrieval

For end-to-end retrieval, you should index the document representations into [FAISS](https://github.com/facebookresearch/faiss).

```sh 
sh index_faiss.sh
```