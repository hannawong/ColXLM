CUDA_VISIBLE_DEVICES="0" \
python -m \
colXLM.train --doc_maxlen 180 --mask-punctuation --bsize 32 --accum 1 --mlm_probability 0.1 \
--triples /path/to/train.tsv \
--prop /path/to/prop/msmarco_info \
--langs "en,fr,es,de,it,pt,nl,sv,pl,ru,ar,zh,ja,ko,hi" \
--root /path/to/ColXLM --experiment WIKI-psg --similarity l2 --run wiki.psg.l2 --maxsteps 2000000