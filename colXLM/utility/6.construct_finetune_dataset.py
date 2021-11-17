import random

top20 = open("../Dataset/Top20.tsv").read().split("\n")
docs = open("../Dataset/Documents.tsv").read().split("\n")
queries = open("../Dataset/queries.tsv").read().split("\n")
OUT = open("../Dataset/finetune_en.tsv",'w')

doc_list = {}
for doc in docs:
    if doc != "":
        doc_id = doc.split("\t")[0]
        doc = doc.split("\t")[1]
        doc_list[doc_id] = doc

doc_len = len(doc_list.keys())

query_list = {}
for query in queries:
    if query != "":
        query_id = query.split("\t")[0]
        query_text = query.split("\t")[1]
        query_list[query_id] = query_text

cnt = 0
print(doc_len)
for top20_ in top20:
    if top20_ is not "":
        cnt += 1
        top20_ = top20_.split("\t")
        query_id = top20_[0]
        doc_id = top20_[1]  ## positive
        #### negative sampling ####
        neg_id = top20_[1]
        while str(neg_id) in top20_:
            neg_id = random.randint(0,doc_len-1)
        print("aaaaa")
        #print("#########",doc_list[str(neg_id)])
        OUT.write(query_list[query_id]+"\t"+doc_list[doc_id]+"\t"+doc_list[str(neg_id)]+"\n")
print(cnt)
