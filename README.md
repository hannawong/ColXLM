# Cross-lingual Information Retrieval Model for Document Search

Hi there! 👋 In this repository, we develop a Cross-lingual Information Retrieval model that support 15 different languages, and it will be used on Yelp search engine after further online experiments. 

## Background
The current search engine of Yelp is based on [NrtSearch](https://engineeringblog.yelp.com/2021/09/nrtsearch-yelps-fast-scalable-and-cost-effective-search-engine.html). However, inverted index-based lexical matching on Lucene-based search engine such as NrtSearch falls short in several key aspects: 
- Lack of understanding of hypernyms, synonyms, and antonyms. For example, *"sneaker"* might match the intent of the query *"running shoes"*, but may not be retrieved.
- Fragility of morphological variants (e.g. *woman* vs. *women*)
- Sensitivity to spelling errors
- Inability to support cross-lingual search

Although Yelp rewrites the query by query expansion and spelling correction before sending it to search engine, the capacity of this method is still limited. Therefore, we intend to add a neural-network-based model trained with large amount of text to complement the lexical serach engine in ad-hoc multilingual retrieval.

## Pretraining Phase

#### Pretraining Tasks
Both [mBERT](https://arxiv.org/pdf/1810.04805.pdf) and [XLM](https://arxiv.org/pdf/1901.07291.pdf) focus on word-level tasks during pretraining (MLM and TLM). The fact that they perform well on word and sentence level tasks but poorly on retrieval tasks suggests that the representations of longer sequences might not be well aligned in cross-lingual LMs. Therefore, we use two pretraining objective specially designed for cross-lingual retrieval tasks:

- 

#### Pretraining Dataset Construction
We use an in-house translation model to translate queries to 15 different languages. 
The ColXLM-15 model includes these languages: en-fr-es-de-it-pt-nl-sv-pl-ru-ar-zh-ja-ko-hi. These abbrievations are represented by [ISO 639-2 Code](https://www.loc.gov/standards/iso639-2/php/code_list.php)

#### Pretraining Details
We continue pretraining our retrieval-oriented language models from the public [mBERT checkpoint](https://huggingface.co/bert-base-multilingual-uncased). Therefore, our cross-lingual LM is implicitly pretrained with three objectives(MLM, QLM, RR). We first train with RR in random order of languange pairs, then train with QLM in random order of language pairs in each iteration. Each epoch contains 32K positive query-document pairs per language pair for each objective. 

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

## Contact us
Zihan Wang: zw2782@columbia.edu

Columbia Database Group: https://cudbg.github.io/
