# Cross-lingual Information Retrieval Model for Document Search

## Train Phase

```sh
CUDA_VISIBLE_DEVICES="0,1,2,3" \
python -m torch.distributed.launch --nproc_per_node=4 -m \
colXLM.train --amp --doc_maxlen 180 --mask-punctuation --bsize 32 --accum 1 \
--triples /data/jiayu_xiao/project/wzh/Cross-lingual-Retrieval-Pretrain-Model/colXLM/Dataset/triples.train.small.tsv \
--root /data/jiayu_xiao/project/wzh/Cross-lingual-Retrieval-Pretrain-Model --experiment MSMARCO-psg --similarity l2 --run msmarco.psg.l2 --maxsteps 1000
```