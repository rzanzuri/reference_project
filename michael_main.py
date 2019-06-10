from keras.models import Sequential
from keras import layers
from keras.layers import Embedding, LSTM, Dense, Activation, Flatten, Bidirectional
from keras.models import model_from_json
import Python_lib.dataSets as dataSets
import Python_lib.vectorsModel as vectorsModel
import Python_lib.named_entity_recognotion as NER
import Python_lib.textHandler as textHandler
import numpy as np
import multiprocessing
import logging
import gensim

import datetime
import sys
from threading import Thread
import os
from os import listdir,mkdir,rmdir
from os.path import isfile, join,isdir
import Python_lib.read_html as read_html
from tika import parser
import re
from Python_lib.textHandler import clean_hebrew_text_from_dir

def dror_task():
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

if __name__ == "__main__":
    start = datetime.datetime.now()
    print("start:", start)
    # dror_task()
    clean_hebrew_text_from_dir("./Data/Test/", "RabannyText.txt")
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