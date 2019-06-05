import os
import zipfile
import bz2
import nltk
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters
from nltk.tokenize import word_tokenize

def split_text_to_sentences(text, max_sen_len = -1):
    punkt_param = PunktParameters()
    abbreviation = ['u.s.a', 'fig','dr', 'vs', 'mr', 'mrs', 'prof', 'inc','i.e', 'a.m', 'acct', 'approx', 'ave', 'b.a' , 'ba' , 'b.o.t', 'bros', 'cf', 'e.g', 'encl', 'etc', 'ft', 'gal', 'p.a', 'p.m', 'sq', 'st', 'blvd', 'cyn', 'ln', 'rd', 'p.s']
    punkt_param.abbrev_types = set(abbreviation)
    tokenizer = PunktSentenceTokenizer(punkt_param)

    sentences = []

    for line in text.splitlines():
        for sentence in tokenizer.tokenize(line):
            if(max_sen_len == -1):
                sentences.append(sentence)
            else:
                for sent in get_sentences_by_max_length(sentence, max_sen_len):
                    sentences.append(sent)
    return sentences 

def get_sentences_by_max_length(sentence, max_sen_len = -1):
    split_sentences = []
    for i, word in enumerate(word_tokenize(sentence)):
        if i % max_sen_len == 0:
            split_sentences.append(word)  
        else:
            split_sentences[int(i/max_sen_len)] += " " + word

    return split_sentences


def get_len_of_longest_sentence(sentences):
    max_len = 0
    for sentence in sentences:
        x = nltk.tokenize.regexp_tokenize(sentence, pattern=r'\w+|\$[\d\.]+|\S+')
        y = len(x)
        if(y > max_len):
            max_len = y
    return max_len

def get_text_file(file, dir_path = ""):
    file_path = os.path.join(dir_path, file)
    if not os.path.isfile(file_path):
        print("ERROR: File not exists!")
        return None
    elif file_path.endswith(".bz2"):
        with bz2.open(file_path, 'rt', encoding='utf-8') as file:
            text = file.read()
    else:
        with open(file_path, encoding='utf-8') as file:
            text = file.read()          
    return text

def get_sentences_from_file(file, dir_path = "", max_len_sent = -1):
    file_path = os.path.join(dir_path, file)
    text = get_text_file(file_path)
    return split_text_to_sentences(text, max_len_sent)


def get_sentences_from_dir(dir_path, max_len_sent = -1):
    sentences = []
    for filename in os.listdir(dir_path): 
        text = get_text_file(filename,dir_path)       
        sentences =  sentences + split_text_to_sentences(text, max_len_sent)
    if not sentences:
            return None
    return sentences

def get_sentences(path,dir_path = "") :
    if(os.path.isdir(path)):
        return get_sentences_from_dir(path)
    elif (os.path.isfile(os.path.join(dir_path,path))):
        return get_sentences_from_file(path, dir_path)
    else:
        print("ERROR: get_sentences")
        return None

def get_temp(dir_path, dest_dir):
    line = ""
    for filename in os.listdir(dir_path): 
        with open(dest_dir + filename + ".lettres", 'w', encoding = 'utf-8') as f:
            text = get_text_file(filename,dir_path)    
            for char in text:
                if(char == "\t"):
                    line += "| "
                elif char == "\n":
                    line += char
                    f.write(line)
                    line = ""
                else:
                    line += char + " "