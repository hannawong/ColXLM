CUDA_VISIBLE_DEVICES="7" \
python -m colXLM.retrieve_faiss \
--checkpoint_path /path/to/checkpoints/colbert.dnn \
--query_doc_path /path/to/queries.tsv \
--submit_path /path/to/submit.tsv \
--index_path /path/to/indexes \
--gold_path /path/to/top1000.tsv \
--faiss_name faiss_l2 \
--batchsize 128 \
--k 1000   