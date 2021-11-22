CUDA_VISIBLE_DEVICES="0" \
python -m colXLM.index_document \
--checkpoint_path "/data/jiayu_xiao/project/wzh/ColXLM/MSMARCO-psg/train.py/msmarco.psg.rronly/checkpoints/colbert.dnn" \
--index_path "/data/jiayu_xiao/project/wzh/ColXLM/colXLM/indexes" \
--doc_path "/data/jiayu_xiao/project/wzh/Documents.tsv"
