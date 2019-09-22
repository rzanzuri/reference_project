import MyLibs.dataSets as dataSets
import MyLibs.vectorsModel as vectorsModel
import MyLibs.named_entity_recognotion as NER
import MyLibs.textHandler as textHandler
import numpy as np
import multiprocessing
import logging
import gensim
import random

import datetime
import sys
from threading import Thread
import os
from os import listdir,mkdir,rmdir
from os.path import isfile, join,isdir
import MyLibs.read_html as read_html
#from tika import parser
import re
from MyLibs.textHandler import clean_hebrew_text_from_dir


def preprocess_sentence(w):
    # w = unicode_to_ascii(w.lower().strip())

    # creating a space between a word and the punctuation following it
    # eg: "he is a boy." => "he is a boy ."
    # Reference:- https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)

    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    if(any("\u0590" <= c <= "\u05EA" for c in w)):
        w = re.sub(r"[^א-ת?.!,¿]+", " ", w)
    else:
        w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)

    w = w.rstrip().strip()

    # adding a start and an end token to the sentence
    # so that the model know when to start and stop predicting.
    w = start_seq + ' ' + w + ' ' + end_seq
    return w
    
def dror_task():
    from keras.models import Sequential
    from keras import layers
    from keras.layers import Embedding, LSTM, Dense, Activation, Flatten, Bidirectional
    from keras.models import model_from_json    
    iters = 6
    min_count = 60
    vec_size = 300 
    win_size = 20
    workers = multiprocessing.cpu_count() 

    #model setup
    vec_model_root_path = "./VecModels"
    curpus_path = "./Data/shuffled_clean_shut/"
    vec_model = vectorsModel.get_model_vectors(curpus_path, vec_model_root_path, win_size, iters, min_count, vec_size, workers)
    write_dict(vec_model, vec_model.wv.vocab, curpus_path + "shuffled_clean_shut_vocab.txt")
    find_similar(curpus_path, vec_model)

def write_dict(vec_model, vocab, file_name):
    with open(file_name, 'w', encoding = 'utf-8') as f:
        for word in vocab:
            f.write(word +  "\n")
            for val in vec_model.wv[word]:
                f.write(str(val) + " ")
            f.write("\n")

def find_similar(curpus_path, vec_model):
    special_words = []
    with open("./Data/shuffled_clean_shut/special words.txt", 'r', encoding = "utf-8") as f:
        for line in f.readlines():
            special_words.append(line.strip("\n"))
    
    for word in special_words:
        file_name = curpus_path + "similarity/" + word.replace("\"","\'\'") + "_most_similar.txt"
        with open(file_name, 'w', encoding = 'utf-8') as f:
            if word in vec_model.wv.vocab:
                most_similar = vec_model.most_similar(positive=[word], topn = 100)
                for sim_line in most_similar:
                    if sim_line[0] in vec_model.wv.vocab:
                        f.write(sim_line[0] + "\n" + str(sim_line[1]) + "\n")
                        for val in vec_model.wv[word]:
                            f.write(str(val) + " ")
                        f.write("\n")
                        
def write_text_to_file_from_url(url, start, amount, website_name, html_part = ["p"]):
    for i in range(start, start + amount):
        try:
            #if isfile(f"./Data/download_pages/{website_name}_{i}.txt"): continue
            html = read_html.read_url_2_text(f"{url}{i}")
            if html == "":
                continue
            text = read_html.parse_html_to_text(html, html_part)
            if len(text) > 10:
                if(not isdir(f"./Data/download_pages/{website_name}/")):
                    mkdir(f"./Data/download_pages/{website_name}/")
                with open(f"./Data/download_pages/{website_name}/{website_name}_from_{start}_to_{start+amount}.txt", "a", encoding="utf-8") as f:
                    f.write(text)
        except:
            print("Unexpected error:", sys.exc_info())

def write_text_to_file_from_url_of_maariv(url, start, amount, website_name):

    try:
        text = ""
        for i in range(start, start + amount):
            try:
                html = read_html.read_url_2_text(f"{url}{i}")
                text += read_html.parse_maariv_html_to_text(html) + "\n\n"
            except:
                print("Unexpected error:", sys.exc_info())
        if len(text) < 10:
            return
        with open(f"./Data/download_pages/{website_name}/{website_name}_{start}-{start+amount-1}.txt", "a", encoding="utf-8") as f:
            f.write(text)                 

    except:
        print("Unexpected error:", sys.exc_info())    

def run_threads(thread_count, start, count, url, name):

    threads = []
    try:
        for i in range(0, thread_count):
            threads.append(Thread(target=write_text_to_file_from_url, args=(url, start + i * count, count, name,["p"])))

        [t.start() for t in threads]
        [t.join() for t in threads]

    except:
        print ("Error: unable to start thread")
  
def create_vec_model():
    from keras.models import Sequential
    from keras import layers
    from keras.layers import Embedding, LSTM, Dense, Activation, Flatten, Bidirectional
    from keras.models import model_from_json
    iters = 25
    min_count = 15 
    vec_size = 300 
    win_size = 12
    workers = multiprocessing.cpu_count() 
    vec_model_root_path = "./VecModels"
    #curpus_path = "./Data/hebrew_data/HebrewTextPart"
    curpus_path = "./Data/english_data/english_te"
    
    vec_model = vectorsModel.get_model_vectors(curpus_path, vec_model_root_path, win_size, iters, min_count, vec_size, workers)

    return vec_model

def download_maariv_pages():
    start_article = 670001
    end_article = 702604
    num_of_sites_in_file = 100
    num_of_thread_in_parallel = 100
    article_index = start_article

    while article_index < end_article:
        threads = []
        i = 0
        while i < num_of_thread_in_parallel:
            threads.append(Thread(target=write_text_to_file_from_url_of_maariv, args=("https://www.maariv.co.il/news/israel/Article-", article_index, num_of_sites_in_file, "maariv")))
            article_index += num_of_sites_in_file
            i += 1
        
        [t.start() for t in threads]
        [t.join() for t in threads]

def write_n_sentences(amount, src_file , dest_file):
    i = 0
    sentences = []
    with open(src_file,'r', encoding = 'utf-8') as f:
        lines = f.readlines()
    
    for j, line in enumerate(lines):
        for sent in line.split('.'):
            if len(sent.split()) > 4  and len(sent.split()) <= 50:
                sentences.append(sent.lstrip().rstrip() + ".")

    with open(dest_file,'w' , encoding = 'utf-8') as f:
        while i < amount:
            idx = random.randint(0,len(sentences) -1)
            f.write(sentences[idx] + "\n")
            sentences.pop(idx)
            i+=1    

def create_rand_sents():
    t1  = Thread(target=write_n_sentences, args=(3000,"./Data/RabannyText/all.txt","./Data/RabannyText/RabannyText_3000_Sen.txt"))
    # t2  = Thread(target=write_n_sentences, args=(3000,"./Data/HebrewText/HebrewText.txt","./Data/HebrewText/HebrewText_3000_Sen.txt"))
    
    t1.start()
    # t2.start()

    t1.join()
    # t2.join()

def create_ans_file(file_path, true_count, false_count, dest_file_path):
    sents = []
    anses = []
    final_sents = []
    final_anses = []
    indexes = list(range(true_count + false_count))
    random.shuffle(indexes)
    with open(file_path, 'r',encoding='utf-8') as f:
        lines = f.readlines()

    for i ,line in enumerate(lines):
        if i % 2 == 1 :
            anses.append(line)
        else:
            sents.append(line)

    for i in range(len(sents)):
        if anses[i][0] == "1" and true_count == 0:
            continue
        elif anses[i][0] == "0" and false_count == 0:
            continue

        final_sents.append(sents[i])
        final_anses.append(anses[i])

        if anses[i][0] == "1":
            true_count -= 1
        else: false_count -= 1

    with open(dest_file_path,'w', encoding='utf-8') as f:
        for i in indexes:
            f.write(final_sents[i])
            f.write(final_anses[i])

def create_ans_for_rabanny_text(file_path):
    with open(file_path,'r',encoding='utf-8') as rf:
        with open(file_path + ".new", 'w', encoding='utf-8') as wf:
            for line in rf.readlines():
                wf.write(line.strip().replace('# ','').replace('$ ', '') + '\n')
                wf.write(line.strip() + '\n')

def hebrew_qualty(vec_model):
    words = []
    words.append(["ש לא"])
    words.append(["ש ל א"])

    words.append(["נקודה ות","נקודות"])
    words.append(["נ ק ו ד ות"])
    words.append(["נקודה ו ת","נקודות"])
    words.append(["נ ק ו ד ו ת"])

    words.append(["ה אחרונים","האחרונים"])
    words.append(["האחרון ים","האחרונים"])
    words.append(["ה אחרון ים","האחרונים"])
    words.append(["ה אחרון י ם","האחרונים"])
    words.append(["ה א ח ר ו נ י ם"])
    words.append(["ה א ח ר ו נ ים"])

    words.append(["ה שנים"])
    words.append(["השנה ים","השנים"])
    words.append(["ה שנה ים","השנים"])
    words.append(["ה ש נ ים"])
    words.append(["ה ש נ י ם"])

    words.append(["כש אכלתי"])
    words.append(["כש אכל תי"])
    words.append(["כ ש אכל תי"])
    words.append(["כש אכל ת י"])
    words.append(["כ ש א כ ל ת י"])
    # words.append([["ש","ל","י"]])
    # words.append([["א","נ","י"]])
    # words.append([["ש","ל"]])
    # words.append([["י","ש"]])
    # words.append([["כ","ן"]])
    # words.append([["א","ת"]])
    # words.append([["ע","ל"]])
    # words.append([["כ","י"]])
    # words.append([["ז","ה"]])
    # words.append([["ל","א"]])
    # words.append([["א","ו"]])
    # words.append([["ה","ם"]])
    # words.append([["כ","ך"]])
    # words.append([["ר","ק"]])
    # words.append([["ש","לא"]])
    # words.append([["ב","כל"]])
    # words.append([["ש","נ","ה","ים"],"שנים"])
    # words.append([["נ","ק","ו","ד","ה","ות"],"נקודות"])
    # words.append([["ה","ו","א"]])
    # words.append([["א","ב","ל"]])
    # words.append([["ה","י","א"]])
    # words.append([["ה","י","ה"]])
    # words.append([["נ","ק","ו","ד","ה"]])
    # words.append([["ה","ק","ב","ו","צ","ה"]])
    # words.append([["ה", "אחרון","ים"],"האחרונים"])
    # words.append([["ה", "אחרון","ה"],"האחרונה"])
    # words.append([["ה", "שנה","ים"],"השנים"])
    # words.append([["טוב", "ה"]])
    # words.append([["חודש", "ים"]])
    # words.append([["ב", "ירושלים"]])
    # words.append([["נקודה","ות"],"נקודות"])
    # words.append([["ב","בית"]])

    for word in words:
        most_similar = vec_model.most_similar(positive=word[0].split(" "), topn = 1000000)
        print("word:", word[0].split(" ") )
        if len(word) == 1:
            full_word = word[0].replace(" ", "")
        else:
            full_word = word[1]
        for i, mo_sim in enumerate(most_similar):
            if mo_sim[0] == full_word:
                print( "number:" ,i ,mo_sim )             
                break
    # words.append([["ש","ל","י"],["שלי"]])
    # words.append([["א","נ","י"],["אני"]])
    # words.append([["ש","ל"],["של"]])
    # words.append([["י","ש"],["יש"]])
    # words.append([["כ","ן"],["כן"]])
    # words.append([["נ","ק","ו","ד","ה"],["נקודה"]])
    # words.append([["טוב", "ה"],["טובה"]])
    # words.append([["נקודה","ות"],["נקודות"]])

    # for word in words:
    #     most_similar = vec_model.most_similar(positive=word[0], topn = 1000000)
    #     print("word:", word[0] )
    #     full_word = word[1]
    #     for i, mo_sim in enumerate(most_similar):
    #         if mo_sim[0] == full_word:
    #             print( "number:" ,i ,mo_sim )
    #             break

def english_qualty(vec_model):
    words = []

    words.append(["un like"])
    words.append(["un l i k e"])
    words.append(["u n like"])
    words.append(["u n l i k e"])

    words.append(["like ly"])
    words.append(["like l y"])
    words.append(["l i k e ly"])
    words.append(["l i k e l y"])

    words.append(["dis like"])
    words.append(["d i s like"])
    words.append(["dis l i k e"])
    words.append(["d i s l i k e"])

    words.append(["anti virus"])
    #words.append(["anti - virus"])
    words.append(["a n t i virus"])
    words.append(["anti v i r u s"])
    words.append(["a n t i v i r u s"])


    words.append(["be ing"])
    words.append(["be i n g"])
    words.append(["b e ing"])
    words.append(["b e i n g"])

    words.append(["include ing","including"])
    words.append(["i n c l u d ing","including"])
    words.append(["include i n g","including"])

    words.append(["return ed"])
    words.append(["re turned"])
    words.append(["re turn ed"])
    words.append(["r e turn ed"])
    words.append(["re turn e d"])
    words.append(["r e turn e d"])
    words.append(["r e t u r n e d"])
    words.append(["re t u r n ed"])

    words.append(["un usual"])
    words.append(["u n usual"])
    words.append(["un u s u a l"])
    words.append(["u n u s u a l"])

    words.append(["worth less"])
    words.append(["w o r t h less"])
    words.append(["worth l e s s"])
    words.append(["w o r t h l e s s"])

    # words.append([["t","h","e"]])
    # words.append([["a","n","d"]])
    # words.append([["b","e"]])
    # words.append([["be","en"]])
    # words.append([["be","ing"]])
    # words.append([["include","ing"],"including"])
    # words.append([["child","ren"]])
    # words.append([["re","turn","ed"]])
    # words.append([["work","ed"]])
    # words.append([["continue","ed"],"continued"])
    # words.append([["continue","d"]])
    # words.append([["do","es"]])
    # words.append([["go","es"]])
    # print(vec_model.wv.vectors.shape[0])
    #words.append([["א","נ","י"]])
    #words.append([["ש","נ","ה","ים"],"שנים"])

    for word in words:
        vector = max_polling(vec_model, word[0].split())
        max_poling_similar = 1 - vector_dist(vector, vec_model.wv[word[0].replace(" ", "")])
        most_similar = vec_model.most_similar(positive=word[0].split(" "), topn = 1000000)
        print("word:", word[0].split(" ") )
        print("Distance with max poling:", max_poling_similar )
        if len(word) == 1:
            full_word = word[0].replace(" ", "")
        else:
            full_word = word[1]
        for i, mo_sim in enumerate(most_similar):
            if mo_sim[0] == full_word:
                print( "number:" ,i ,mo_sim )             
                break
def vector_dist(vec1, vec2):
    dst = spatial.distance.cosine(vec1, vec2)
    return dst


def max_polling(model, broken_word):
    new_vector = np.zeros(300)
    for i in range(300):
        #maximum = None
        maximum = 0
        for partial in broken_word:
            #if maximum == None or maximum < model.wv[partial][i]:
            #    maximum = model.wv[partial][i]
            maximum += model.wv[partial][i]
        new_vector[i] = maximum
    return new_vector

def get_distance_of_letters(vec_model, word):
    most_similar = vec_model.most_similar(positive=list(word), topn = 1000000)
    full_word = word

    for i, mo_sim in enumerate(most_similar):
        if mo_sim[0] == full_word:
            print( "number:" ,i + 1 ,mo_sim )          
            return( i + 1, mo_sim[1])

if __name__ == "__main__":
    start = datetime.datetime.now()
    print("start:", start)

    with open("./Data/RabannyText/RabannyText.ans",'r', encoding='utf-8') as rf:
        with open("./Data/RabannyText/RabannyText.ans.true_flase",'w', encoding='utf-8') as wf:
            lines = rf.readlines()
            for i, line in enumerate(lines):
                if i%2:
                    if '<start_ref>' in line:
                        wf.write("1\n")
                    else:
                        wf.write("0\n")
                else:
                    wf.write(line)    
                    
    with open("./Data/spa-eng/spa.txt",'r', encoding='utf-8') as rf:
        with open("./Data/spa-eng/spa.ans",'w', encoding='utf-8') as wf:
            lines = rf.readlines()
            for line in lines:
                line = line.split('\t')
                wf.write(line[0].strip().rstrip() + "\n")
                wf.write(line[1].strip().rstrip() + "\n")
    # create_rand_sents()
    # create_ans_for_rabanny_text('./Data/RabannyText/RabannyText_3000_Sen.txt')
    # base_path = "C:\\hebrewNER-1.0.1-Win\\bin\\"
    # with open("./Data/HebrewText/HebrewText_3000_Sen.txt",'r',encoding='utf-8') as f:
    #     text = f.readlines()

    # with open("./Data/HebrewText/HebrewText.ans",'w',encoding='utf-8') as ans_file:
    #     for i, sent in enumerate(text):
    #         file_name = "testFiles\\" + str(i) + ".maxent"
    #         with open(base_path + file_name,'r') as f:
    #             line = f.read()
    #             if "PERSON" in line or "LOCATION" in line or "ORGANIZATION" in line:
    #                 ans  = "1"
    #             else:
    #                 ans = "0"
    #         ans_file.write(sent + ans + "\n")
        
    
    
    # create_rand_sents()
    # vec_model = create_vec_model()
    # most_similar = vec_model.most_similar(positive=["נוסף", "ת"], topn = 10)
    # for sim_line in most_similar:
    #     if sim_line[0] in vec_model.wv.vocab:
    #         print(sim_line[0] + "\n" + str(sim_line[1]) + "\n")
                


    
    # create_ans_file("./Data/‏‏HebrewTextForNER/HebrewTextForNER_3000sen.ans", 550, 550 , "./Data/‏‏HebrewTextForNER/HebrewTextForNER.ans")
    # create_rand_sents()    
    # create_vec_model()
    # dror_task()
    # clean_hebrew_text_from_dir("./Data/RabannyText/", "RabannyText.txt")
    # download_maariv_pages()

    # raw = parser.from_file('./Data/parasha-bereshit.pdf')
    # text = raw['content']
    # with open("./Data/parasha-bereshit.txt", 'w', encoding="utf-8") as f:
    #     f.write(text)

    finish = datetime.datetime.now()
    print("end:", finish)
    print("total:", finish - start)

# import urllib.request as urllib2
# from html.parser import HTMLParser
# from bs4 import BeautifulSoup
# import re


# def cleanhtml(raw_html):
#   cleanr = re.compile('<.*?>')
#   cleantext = re.sub(cleanr, '', raw_html)
#   return cleantext

# parser = HTMLParser()
# #html_page = urllib2.urlopen("https://www.nytimes.com/")
# new_page = []
# with open("./Data/Online/f_00711.html", encoding = "ut") as page:
#     lines = page.readlines()
#     for line in lines:
#         new_line = cleanhtml(line)
#         if new_line != "" and new_line != "\n" and new_line != "\t\n":
#             new_page.append(new_line)
#             print("line:" , line)
#             print("new line:", new_line)

#parser.feed(str(html_page.read()))
#print("Start tags", parser.lsStartTags)
#print("End tags", parser.lsEndTags)
#print("Start End tags", parser.lsStartEndTags)
#print("Comments", parser.lsComments)
#start = parser.get_starttag_text()
#print(start)
#a="אעהגכ"


# import gensim
# import random
# import datetime

# def do_shuf(arr):
#     random.shuffle(arr)
#     return arr

# a = ['1','2','3','4','5','6','7']
# print(do_shuf(a))
# print(a)
# print(do_shuf(a))
# print(a)
# print(datetime.datetime.now)


# sentences = gensim.models.word2vec.LineSentence("./Data/clean_shut/clean_shut.txt")
# model = gensim.models.Word2Vec.load("./VecModels/clean_shu.vm")
# x = model.wv['שאלה']
# print("fisrt:")
# print(x)
# model.train(sentences, total_examples=1, epochs=1)
# y = model.wv['שאלה']
# print("sec:")
# print(y)
# print("eq:")

# print(x == y)

# print("if:")
# if x == y:
#     print("if eq")
# else:
#     print("not if eq")