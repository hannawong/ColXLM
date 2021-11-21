CUDA_VISIBLE_DEVICES="0" \
python -m \
colXLM.train --doc_maxlen 180 --mask-punctuation --bsize 12 --accum 1 --mlm_probability 0.1 \
--triples /data/jiayu_xiao/my_data/Dataset \
--langs "zh,fr" \
--root /data/jiayu_xiao/project/wzh/ColXLM --experiment MSMARCO-psg --similarity l2 --run msmarco.psg.rronly --maxsteps 10000 