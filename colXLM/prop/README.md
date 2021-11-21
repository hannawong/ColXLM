# PROP

PROP, **P**re-training with **R**epresentative w**O**rds **P**rediction, is a new pre-training method tailored for ad-hoc retrieval. The paper can be found [here](https://arxiv.org/pdf/2010.10137.pdf)

### Preprocess data

Run the following command to generate files:

`corpus_df_file.json: {word: document tf}`
`doc_tf_file.json: {doc_id, doc_tf, doc_word_num}, one document per line`
`corpus_tf_file.json: {word: corpus tf}`
`info_file.json: {total_doc_num, total_word_num, average_doc_word_num}`
`stem2pos_file.json: {stem: {word: count}}`
`preprocessed_data: {docid, bert_tokenized_doc_text}`

```sh
python -m prop.preprocessing_data \
    --corpus_name msmarco \
    --data_file /data/jiayu_xiao/my_data/Dataset/prop/triples.train.small.tsv \
    --do_lower_case \
    --stem \
    --output_dir /data/jiayu_xiao/my_data/Dataset/prop/msmarco_info
```
### Generate representive word sets

you can tune the hyperparameter `possion_lambda` to match with your target dataset (the average of query length):

```sh
    python -m prop.multiprocessing_generate_word_sets \
        --corpus_info_dir /data/jiayu_xiao/my_data/Dataset/prop/msmarco_info  \
        --output_dir /data/jiayu_xiao/my_data/Dataset/prop/msmarco_info \
        --epochs_to_generate 1 \
        --rop_num_per_doc 1 \
        --num_workers 1 \
        --epochs_to_generate 1 \
        --possion_lambda 8
```

### Generate training instances

```sh
python -m prop.multiprocessing_generate_pairwise_instances \
    --train_corpus /data/jiayu_xiao/my_data/Dataset/prop/msmarco_info \
    --output_dir /data/jiayu_xiao/my_data/Dataset/prop/msmarco_info/train \
    --do_lower_case \
    --rop_num_per_doc 1 \
    --epochs_to_generate 1 \
    --max_seq_len 512 \
    --temp_dir ./  \
    --num_workers 1
```