CUDA_VISIBLE_DEVICES="0" \
python -m \
colXLM.train --doc_maxlen 180 --mask-punctuation --bsize 24 --accum 1 \
--triples /data/jiayu_xiao/project/wzh/ColXLM/colXLM/Dataset/triples.train.small.tsv \
--root /data/jiayu_xiao/project/wzh/ColXLM --experiment MSMARCO-psg --similarity l2 --run msmarco.psg.l2 --maxsteps 10001 
#--checkpoint /data/jiayu_xiao/project/wzh/ColXLM/MSMARCO-psg/train.py/msmarco.psg.l2/checkpoints/colbert.dnn --resume