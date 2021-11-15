import re
import html
from urllib import parse
import requests
import time

GOOGLE_TRANSLATE_URL = 'http://translate.google.cn/m?q=%s&tl=%s&sl=%s'
TRAIN_PATH = "/data1/jiayu_xiao/project/datasets/Dataset/triples.train.small.tsv"
OUT_TRAIN_PATH = "/data1/jiayu_xiao/project/datasets/Dataset/triples.train.fr.tsv"

def translate(text, to_language="auto", text_language="auto"):

    text = parse.quote(text)
    url = GOOGLE_TRANSLATE_URL % (text,to_language,text_language)
    response = requests.get(url)
    data = response.text
    expr = r'(?s)class="(?:t0|result-container)">(.*?)<'
    result = re.findall(expr, data)
    if (len(result) == 0):
        return ""

    return html.unescape(result[0])

Train = open(TRAIN_PATH,"r")
OUT_train = open(OUT_TRAIN_PATH,"w")
for i in range(10000):
    print(i)
    query_tot = ""
    doc1_list = []
    doc2_list = []
    for j in range(32):
        line = Train.readline().split("\t")
        query_tot += line[0]+'\n'
        doc1 = line[1]
        doc2 = line[2]
        doc1_list.append(doc1)
        doc2_list.append(doc2)

    query_fr = translate(query_tot, "fr","en").split("\n")
    time.sleep(1)

    for j in range(32):
        OUT_train.write(query_fr[j]+"\t"+doc1_list[j]+"\t"+doc2_list[j])

#print(translate("how many calories a day are lost breastfeeding ", "fr","en")) #汉语转英语