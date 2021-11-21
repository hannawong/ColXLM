msmarco_path = open("/data/jiayu_xiao/my_data/Dataset/triples.train.small.tsv")
output = open("/data/jiayu_xiao/my_data/Dataset/prop/triples.train.small.tsv","w")
for i in range(100000):
    output.write(msmarco_path.readline().split("\t")[1]+"\n")