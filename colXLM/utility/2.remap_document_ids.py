top20 = open("../Dataset/top20.tsv").read().split("\n")
docs = open("../Dataset/documents.tsv").read().split("\n")

pid2new = {}
OUT_docs = open("../Dataset/Documents.tsv","w")

for i, doc in enumerate(docs[:-1]):
    doc = doc.split("\t")
    doc_id = int(doc[0])
    pid2new[doc_id] = i
    OUT_docs.write(str(i)+"\t"+doc[1]+"\n")

out_top20 = open("../Dataset/Top20.tsv","w")

for top in top20[:-1]:
    top = top.split("\t")
    for i in range(1,21):
        top[i] = str(pid2new[int(top[i])])
    top = "\t".join(top)[:-1]
    out_top20.write(top+"\n")


