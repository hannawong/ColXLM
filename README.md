# Cross-lingual Information Retrieval Model for Document Search

Hi there! 👋 In this repository, we develop a Cross-lingual Information Retrieval model that support 15 different languages, and it will be used on Yelp search engine after further online experiments. 

<p align="center">
  <img align="center" src="fig/yelp.PNG" />
</p>
<p align="center">
  <b>Figure 1:</b> Yelp's search interface
</p>

## Background
The current search engine of Yelp is based on [NrtSearch](https://engineeringblog.yelp.com/2021/09/nrtsearch-yelps-fast-scalable-and-cost-effective-search-engine.html). However, inverted index-based lexical matching on Lucene-based search engine such as NrtSearch falls short in several key aspects: 
- Lack of understanding of hypernyms, synonyms, and antonyms. For example, *"sneaker"* might match the intent of the query *"running shoes"*, but may not be retrieved.
- Fragility of morphological variants (e.g. *woman* vs. *women*)
- Sensitivity to spelling errors
- Inability to support cross-lingual search

Although Yelp rewrites the query by query expansion and spelling correction before sending it to search engine, the capacity of this method is still limited. Therefore, we intend to add a neural-network-based model trained with large amount of text to complement the lexical search engine in ad-hoc multilingual retrieval.

## Pretraining Phase

#### Pretraining Tasks
Both [mBERT](https://arxiv.org/pdf/1810.04805.pdf) and [XLM](https://arxiv.org/pdf/1901.07291.pdf) have shown great sucess when fine-tuned on downstream tasks. However, pre-training objectives tailored for ad-hoc information retrieval task have not been well explored. In this repository, we use three pretraining objective specially designed for multilingual retrieval tasks:

- Query Language Modeling Task (QLM)

Mask some query tokens and ask the model to predict the masked tokens based on query contexts and full relevant document.
- Relevance Ranking Task (RR)

Given a query and several documents, the model is asked to rank these documents based on levels of relevance. 
- Representative wOrds Prediction (ROP)

//TODO

#### Model Architecture

In order to be both efficient and effective, we use [ColBERT](https://arxiv.org/pdf/2004.12832.pdf) as backbone. ColBERT relies on fine-grained contextual late interaction to enable scalable BERT-based search over large text collections in tens of milliseconds.

<p align="center">
  <img align="center" src="fig/ColBERT-Framework-MaxSim-W370px.png" />
</p>
<p align="center">
  <b>Figure 2:</b> ColBERT's late interaction structure
</p>


#### Pretraining Dataset Construction
We use [multiligual Wiki](https://dumps.wikimedia.org/) as pretraining dataset, and our approach is conceptually similar to the Inverse Cloze Task (ICT), where one sentence is sampled from a Wiki paragraph as query, and the rest of the paragraph is treated as document. We also use `triples.train.small.tar.gz` from [MSMARCO PASSAGE RANKING DATASET](https://github.com/microsoft/MSMARCO-Passage-Ranking) as training corpus, and use an in-house translation model to translate it into 15 languages. 
The ColXLM-15 model includes these languages: en-fr-es-de-it-pt-nl-sv-pl-ru-ar-zh-ja-ko-hi, represented by [ISO 639-2 Code](https://www.loc.gov/standards/iso639-2/php/code_list.php)

#### Pretraining Details
We continue pretraining our retrieval-oriented language models from the public [mBERT checkpoint](https://huggingface.co/bert-base-multilingual-uncased). Therefore, our cross-lingual LM is implicitly pretrained with four objectives (MLM, QLM, RR, ROP). We first train with QLM in random order of languange pairs, then train with RR and ROP in random order of language pairs in each iteration. Each epoch contains 320K query-document pairs per language pair for each objective. 

In order to train the model, you need to run `train.sh`:

```sh
CUDA_VISIBLE_DEVICES="0" \
python -m \
colXLM.train --doc_maxlen 180 --mask-punctuation --bsize 32 --accum 1 --mlm_probability 0.1 \
--triples /path/to/train.tsv \
--langs "en,fr,es,de,it,pt,nl,sv,pl,ru,ar,zh,ja,ko,hi" \
--root /path/to/ColXLM --experiment WIKI-psg --similarity l2 --run wiki.psg.l2 --maxsteps 10000
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
