CUDA_VISIBLE_DEVICES="1" \
python -m \
colXLM.train --doc_maxlen 180 --mask-punctuation --bsize 32 --accum 1 \
--triples /data/jiayu_xiao/project/wzh/ColXLM/colXLM/Dataset/triples.train.tsv \
--root /data/jiayu_xiao/project/wzh/ColXLM --experiment MSMARCO-psg --similarity l2 --run msmarco.psg.l2 --maxsteps 1001