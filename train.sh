CUDA_VISIBLE_DEVICES="0" \
python -m \
colXLM.train --doc_maxlen 180 --mask-punctuation --bsize 12 --accum 1 \
--triples /data/jiayu_xiao/my_data/Dataset/triples.train.small.tsv \
--root /data/jiayu_xiao/project/wzh/ColXLM --experiment MSMARCO-psg --similarity l2 --run msmarco.psg.rronly --maxsteps 4000000 
#--checkpoint /data/jiayu_xiao/project/wzh/ColXLM/MSMARCO-psg/train.py/msmarco.psg.l2/checkpoints/colbert.dnn --resume