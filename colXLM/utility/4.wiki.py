######################## construct wiki dataset #############################

import pickle as pkl
import random
from nltk.data import load

tokenizer = load("/data1/jiayu_xiao/project/wzh/gistmail/nltk_data/tokenizers/punkt/german.pickle")


max_pairs = 50000

with open("/data1/jiayu_xiao/project/datasets/Dataset/ende/de_np.pkl", 'rb') as fin:
    source_np = pkl.load(fin)    

with open("/data1/jiayu_xiao/project/datasets/Dataset/ende/en_np.pkl", 'rb') as fin:
    target_np = pkl.load(fin)

max_pairs = min(max_pairs, len(source_np))
        
# sample some positive pairs
        
source2target = {k: v['p'] for k, v in source_np.items()}
source_to_keep = set(random.sample(list(source_np.keys()), max_pairs))
target_to_keep = set([source2target[x] for x in source_to_keep])
        
source_np = {k: v for k, v in source_np.items() if k in source_to_keep}
target_np = {k: v for k, v in target_np.items() if k in target_to_keep}

# we have to consider negative documents, so we cannot use *_to_keep
        
source_docs, target_docs = source_to_keep, target_to_keep
for k, v in source_np.items():
    target_docs.add(v["p"])
    target_docs.update(v["n"])

for k, v in target_np.items():
    source_docs.add(v["p"])
    source_docs.update(v["n"])
        
with open("/data1/jiayu_xiao/project/datasets/Dataset/ende/de_text.pkl", 'rb') as fin:
    source_text = pkl.load(fin)
    source_text = {k: v for k, v in source_text.items() if k in source_docs}
with open("/data1/jiayu_xiao/project/datasets/Dataset/ende/en_text.pkl", 'rb') as fin:
    target_text = pkl.load(fin)
    target_text = {k: v for k, v in target_text.items() if k in target_docs}

print(f"# parallel sections pairs: {len(source_np)}")

flatten_parallel = list(source_np.keys())
all_src_sections = list(source_text.keys())
all_tgt_sections = list(target_text.keys())

num_neg = 1  ### how many negative to sample for a positive sample


def get_item(idx):
    first_segment_ids, second_segment_ids, queries, documents, y = [], [], [], [], []

    src_id = flatten_parallel[idx]
    tgt_id = source_np[src_id]["p"]
    note = 0
        
    # randomly switch source and target
        
    if random.random() < 1:
        # normal order
        note = 0
        first_segment_id, second_segment_id = src_id, tgt_id
        first_segment_ids.append(first_segment_id)
        second_segment_ids.append(second_segment_id)
        first_text = source_text[first_segment_id]
        second_text = target_text[second_segment_id]
        query = random.choice(tokenizer.tokenize(first_text))
        queries.append(query); documents.append(second_text); y.append(1)
        neg_second_segment_ids = []
        hard_second_segment_ids = source_np[first_segment_id]["n"]
        sampling_prob = (3/4) ** len(hard_second_segment_ids) # the probability of sampling an "easy" negative document
        while len(neg_second_segment_ids) < num_neg:
            if random.random() < sampling_prob:
                neg_second_segment_ids.append(random.choice(all_tgt_sections))
            else:
                neg_second_segment_ids.append(random.choice(hard_second_segment_ids))
        for neg_second_segment_id in neg_second_segment_ids:
            first_segment_ids.append(first_segment_id)
            second_segment_ids.append(neg_second_segment_id)
            queries.append(query)
            documents.append(target_text[neg_second_segment_id])
            y.append(0)
        return queries,documents,y
        
        return note, first_segment_ids, second_segment_ids, queries, documents, y

OUT = open("/data1/jiayu_xiao/project/datasets/Dataset/wiki_de.tsv","w")
for i in range(0,len(flatten_parallel)):
    query, document, y = get_item(i)

    OUT.write(query[0]+"\t"+document[0]+"\t"+document[1]+"\n")
