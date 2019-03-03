import numpy as np
import my_py_lib.textHandler as textHandler
from os import listdir
from os.path import basename , dirname, exists, join , isfile, isdir
from my_py_lib.NER import get_ner_sentence
from nltk.tokenize import regexp_tokenize

def get_data_set(sentences, model, vec_len):

    max_sen_len = textHandler.get_len_of_longest_sentence(sentences)
    data = np.zeros((len(sentences), max_sen_len, vec_len))

    for i , sent in  enumerate(sentences):
        for j ,word in enumerate(regexp_tokenize(sent, pattern=r'\w+|\$[\d\.]+|\S+')):
            if word in model.wv.vocab:
                data[i,j] = model.wv[word]
    return data

def get_sentences_and_answers_by_dir_path(dir_path, get_answer_func, save = 1 ,dest_dir = "", limit = -1):
    sentences = []
    answers = []

    if(get_answer_func is None):
        print("ERROR: Must to pass get_answer_func as an argument")
        return None, None
    
    if dir_path == "":
        print("ERROR: File does'r exists")
        return None, None 

    if(dest_dir == ""):
        dest_dir = dir_path

    dest_file = open(dest_dir + "text.ans" ,'w', encoding='utf-8')
    
    for i, sentence in enumerate(textHandler.get_sentences_from_dir(dir_path)):
        if(limit != -1 and i >= limit): break
        answer = get_answer_func(sentence)
        if(answer != ""):
            answers.append(answer)
            sentences.append(sentence)            
            if(save):
                dest_file.write(sentence + "\n")
                dest_file.write(answer + "\n")
    if(save):
        dest_file.close()
    return sentences, answers

def get_sentences_and_answers_by_file_path(file_path, get_answer_func, save = 1 ,dest_dir = "", limit = -1):
    sentences = []
    answers = []

    if(get_answer_func is None):
        print("ERROR: Must to pass get_answer_func as an argument")
        return None, None
    if not exists(file_path):
        print("ERROR: File does'r exists")
        return None, None 
       
    if dest_dir == "":
        dest_dir = dirname(file_path) + "\\"

    dest_file = open(dest_dir + basename(file_path) + ".ans" ,'w')
    
    for i, sentence in enumerate(textHandler.get_sentences_from_file(file_path)):
        if(limit != -1 and i >= limit): break
        answer = get_answer_func(sentence)
        if(answer != ""):
            answers.append(answer)
            sentences.append(sentence)            
            if(save):
                dest_file.write(sentence + "\n")
                dest_file.write(answer + "\n")
    
    if(save):
        dest_file.close()

    return sentences, answers    

def get_sentences_and_answers_from_existing_file(file_path, limit = -1):
    sentences = []
    answers = []
    limit *= 2
    
    for i, sentence in enumerate(textHandler.get_sentences_from_file(file_path)):
        if(limit != -2 and i >= limit): break
        if(i%2 == 0):
            sentences.append(sentence)
        else:
            answers.append(sentence)

    return sentences, answers    

def get_sentences_and_answers(path, get_answer_func = None, save = 1 ,dest_dir = "", limit = -1):
    if exists(path) and isfile(path):
        return get_sentences_and_answers_from_existing_file(path, limit)
    elif isfile(path):
        return get_sentences_and_answers_by_file_path(path,get_answer_func,save,dest_dir, limit)
    elif isdir(path):
        return get_sentences_and_answers_by_dir_path(path,get_answer_func,save,dest_dir, limit)
    else:
        print("Error!")
        return None, None

