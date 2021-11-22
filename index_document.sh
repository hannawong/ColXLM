CUDA_VISIBLE_DEVICES="0" \
python -m colXLM.index_document \
--checkpoint_path /path/to/checkpoints/colbert.dnn \
--index_path /path/to/indexes \
--doc_path /path/to/documents.tsv
