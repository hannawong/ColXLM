CUDA_VISIBLE_DEVICES="1" \
python -m colXLM.index_document \
--checkpoint_path "/data/jiayu_xiao/project/wzh/ColXLM/MSMARCO-psg/train.py/msmarco.psg.l2/checkpoints/colbert.dnn"\
--index_path "/data/jiayu_xiao/project/wzh/ColXLM/colXLM/indexes"\
----doc_path "./colXLM/Dataset/documents.tsv"
