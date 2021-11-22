CUDA_VISIBLE_DEVICES="0" \
python -m colXLM.index_faiss \
--dim 128 --index_path /path/to/indexes --faiss_name 'faiss_l2' 