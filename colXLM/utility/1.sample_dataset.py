
#######      Get 10000 queries from MSMACRO       #######
collection = open("/data/jiayu_xiao/project/wzh/ColBERT_/collection.tsv")
query = open("/data/jiayu_xiao/project/wzh/ColBERT_/queries.dev.tsv")
top1000 = open('/data/jiayu_xiao/project/wzh/ColBERT_/top1000.dev')


qid2query = {}
q2p = {}
for i in range(10000):
    line = query.readline()
    qid = line.split("\t")[0]  
    que = line.split("\t")[1]
    q2p[qid] = []
    qid2query[qid] = que

for i in range(6000000):
    match = top1000.readline().split("\t")
    qid = match[0]
    pid = match[1]
    query = match[2]
    passage = match[3]
    if qid in q2p.keys() and len(q2p[qid]) < 20:
        q2p[qid].append(pid)

pid_list = set()
top20 = open("../Dataset/top20.tsv","w")
out_query = open("../Dataset/queries.tsv","w")
out_docs = open("../Dataset/documents.tsv","w")

for qid in q2p.keys():
    if(len(q2p[qid])==20):
        line = qid+"\t"
        for i in q2p[qid]:
            line += i+"\t"
            pid_list.add(i)

        top20.write(line+"\n")
        out_query.write(qid+"\t"+qid2query[qid])

print(list(pid_list))
Collection = collection.read().split("\n")
print(len(Collection))

for i in pid_list:
    document = Collection[int(i)]
    print(document)
    out_docs.write(document+"\n")

