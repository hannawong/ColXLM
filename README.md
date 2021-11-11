# Cross-lingual Information Retrieval Model for Document Search

## Train Phase

```sh
CUDA_VISIBLE_DEVICES="1" \
python -m \
colXLM.train --doc_maxlen 180 --mask-punctuation --bsize 32 --accum 1 \
--triples /data/jiayu_xiao/project/wzh/ColBERT_/triples.train.small.tsv \
--root /data/jiayu_xiao/project/wzh/ColXLM --experiment MSMARCO-psg --similarity l2 --run msmarco.psg.l2 --maxsteps 1000
```

## indexing document 

```sh
CUDA_VISIBLE_DEVICES="1" \
python -m \
colXLM.index --doc_maxlen 180 --mask-punctuation --bsize 256 \
--checkpoint /data/jiayu_xiao/project/wzh/ColXLM/MSMARCO-psg/train.py/msmarco.psg.l2/checkpoints/colbert.dnn \
--collection /data/jiayu_xiao/project/wzh/ColXLM/colXLM/Dataset/Documents.tsv \
--index_root /data/jiayu_xiao/project/wzh/ColXLM/colXLM/indexing/indexes --index_name MSMARCO.L2.32x200k \
--root /data/jiayu_xiao/project/wzh/ColXLM --experiment MSMARCO-psg
```

## Faiss Indexing for retrieval

python -m colXLM.index_faiss \
--index_root /data/jiayu_xiao/project/wzh/ColXLM/colXLM/indexing/indexes --index_name MSMARCO.L2.32x200k \
--partitions 32768 --sample 0.3 \
--root /data/jiayu_xiao/project/wzh/ColXLM --experiment MSMARCO-psg