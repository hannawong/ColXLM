import re
import html
from urllib import parse
import requests
import time

GOOGLE_TRANSLATE_URL = 'http://translate.google.cn/m?q=%s&tl=%s&sl=%s'
TRAIN_PATH = "/data/jiayu_xiao/my_data/Dataset/Documents.tsv"
OUT_TRAIN_PATH = "/data/jiayu_xiao/my_data/Documents.fr.tsv"

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
OUT_train = open(OUT_TRAIN_PATH,"a")

for i in range(0,8819):
    line = Train.readline()

for i in range(11979):
    try:
        print(i)
        query_tot = ""
        doc1_list = ""
        doc2_list = ""
        for j in range(1):
            line = Train.readline().split("\t")
            query_tot += line[0]+'\n'
            doc1_list += line[1]+"\n"

        query_fr = translate(query_tot, "fr","en").split("\n")
        doc1_fr = translate(doc1_list,"fr","en").split("\n")
        #doc2_fr = translate(doc2_list,"fr","en").split("\n")
        print(doc1_fr[0])

        for j in range(1):
            OUT_train.write(query_fr[j]+"\t"+doc1_fr[j]+"\n")
    except:
        print("no")

#print(translate("how many calories a day are lost breastfeeding ", "fr","en")) #汉语转英语