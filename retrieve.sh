CUDA_VISIBLE_DEVICES="1" \
python -m colXLM.retrieve_faiss \
--checkpoint_path /data/jiayu_xiao/project/wzh/ColXLM/MSMARCO-psg/train.py/msmarco.psg.l2/checkpoints/colbert.dnn \
--index_path /data/jiayu_xiao/project/wzh/ColXLM/colXLM/indexes \
--faiss_name faiss_l2 \
--k 20