import os
import zipfile
import bz2
import nltk


def split_text_to_sentences(text):
    sentences = []
    for line in text.splitlines():
        for sentence in nltk.sent_tokenize(line):
            sentences.append(sentence)
    return sentences 

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

def get_sentences_from_file(file, dir_path = ""):
    file_path = os.path.join(dir_path, file)
    text = get_text_file(file_path)
    return split_text_to_sentences(text)


def get_sentences_from_dir(dir_path):
    sentences = []
    for filename in os.listdir(dir_path): 
        text = get_text_file(filename,dir_path)       
        sentences =  sentences + split_text_to_sentences(text)
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
