# Cross-lingual Information Retrieval Model for Document Search

## Train Phase

```sh
CUDA_VISIBLE_DEVICES="0" \
python -m \
colXLM.train --doc_maxlen 180 --mask-punctuation --bsize 32 --accum 1 \
--triples /data/jiayu_xiao/project/wzh/ColBERT_/triples.train.small.tsv \
--root /data/jiayu_xiao/project/wzh/ColXLM --experiment MSMARCO-psg --similarity l2 --run msmarco.psg.l2 --maxsteps 1000
```

## indexing document 

CUDA_VISIBLE_DEVICES="1" \
python -m \
colXLM.index --doc_maxlen 180 --mask-punctuation --bsize 256 \
--checkpoint /data/jiayu_xiao/project/wzh/ColXLM/MSMARCO-psg/train.py/msmarco.psg.l2/checkpoints/colbert.dnn \
--collection /data/jiayu_xiao/project/wzh/ColXLM/colXLM/Dataset/documents.tsv \
--index_root /data/jiayu_xiao/project/wzh/ColXLM/colXLM/indexing/indexes --index_name MSMARCO.L2.32x200k \
--root /data/jiayu_xiao/project/wzh/ColXLM --experiment MSMARCO-psg