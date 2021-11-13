import re
import html
from urllib import parse
import requests

GOOGLE_TRANSLATE_URL = 'http://translate.google.cn/m?q=%s&tl=%s&sl=%s'
TRAIN_PATH = "/data/jiayu_xiao/project/wzh/ColBERT_/triples.train.small.tsv"
OUT_TRAIN_PATH = "/data/jiayu_xiao/project/wzh/ColXLM/colXLM/Dataset/triples.train.tsv"

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
for i in range(100000):
    print(i)
    line = Train.readline().split("\t")
    query = line[0]
    doc1 = line[1]
    doc2 = line[2]
    query_fr = translate(query, "fr","en")
    print(query_fr)
    OUT_train.write(query_fr+"\t"+doc1+"\t"+doc2)

#print(translate("how many calories a day are lost breastfeeding ", "fr","en")) #汉语转英语