CUDA_VISIBLE_DEVICES="2" \
python -m colXLM.retrieve_faiss \
--checkpoint_path /data/jiayu_xiao/project/wzh/ColXLM/MSMARCO-psg/train.py/msmarco.psg.l2/checkpoints/colbert.dnn \
--query_doc_path /data/jiayu_xiao/project/wzh/queries.tsv \
--submit_path /data/jiayu_xiao/project/wzh/ColXLM/colXLM/submit.tsv \
--index_path /data/jiayu_xiao/project/wzh/ColXLM/colXLM/indexes \
--gold_path /data/jiayu_xiao/project/wzh/Top20.tsv \
--faiss_name faiss_l2 \
--batchsize 128 \
--k 20                                              