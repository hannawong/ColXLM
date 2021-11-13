# Cross-lingual Information Retrieval Model for Document Search

Hi there! 👋 In this repository, we develop a Cross-lingual Information Retrieval model that support 15 different languages, and it will be used on Yelp search engine after further online experiments. 

## Background
The current search engine of Yelp is based on [NrtSearch](https://engineeringblog.yelp.com/2021/09/nrtsearch-yelps-fast-scalable-and-cost-effective-search-engine.html), a Lucene-based search engine. However, inverted index-based lexical matching falls short in several key aspects: 
- Lack of understanding of hypernyms, synonyms, and antonyms. For example, *"sneaker"* might match the intent of the query *"running shoes"*, but may not be retrieved.
- Fragility of morphological variants (e.g. *woman* vs. *women*)
- Sensitivity to spelling errors
- Doesn't support cross-lingual search

Although Yelp rewrites the query by query expansion and spelling correction before sending it to search engine, the capacity of this method is still limited. Therefore, we intend to add a neural-network-based model trained with large amount of text to complement the lexical serach engine in ad-hoc multilingual retrieval.

## Pretraining Phase

We continue pretraining our retrieval-oriented language models from the public mBERT checkpoint. Therefore, our cross-lingual LM is implicitly pretrained with three objectives(MLM, TLM, RR)

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
