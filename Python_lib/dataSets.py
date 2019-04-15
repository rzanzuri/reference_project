import numpy as np
import my_py_lib.textHandler as textHandler
from os import listdir
from os.path import basename , dirname, exists, join , isfile, isdir
from my_py_lib.NER import get_ner_sentence
from nltk.tokenize import regexp_tokenize

def get_data_set(sentences, model, vec_len, by_char = 0, size = 0):
    if by_char:
        max_sen_len = size if size != 0 else max([len(snt) for snt in sentences])
        data = np.zeros((len(sentences), max_sen_len, vec_len))

        for i , sent in  enumerate(sentences):
            for j ,cahr in enumerate(sent):
                if cahr in model.wv.vocab:
                    data[i,j] = model.wv[cahr]                     
        return data

    max_sen_len = textHandler.get_len_of_longest_sentence(sentences)
    data = np.zeros((len(sentences), max_sen_len, vec_len))
    for i , sent in  enumerate(sentences):
        for j ,word in enumerate(regexp_tokenize(sent, pattern=r'\w+|\$[\d\.]+|\S+')):
            if word in model.wv.vocab:
                data[i,j] = model.wv[word]
            elif word == "\t":
                data[i,j] = np.zeros(vec_len)
                data[i,j ,5] = 1
            elif word == "\n":
                data[i,j] = np.zeros(vec_len)
                data[i,j ,10] = 1
            elif word == ".":
                data[i,j] = np.zeros(vec_len)
                data[i,j ,15] = 1                            
                

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

    if save:
        dest_dir = dir_path if dest_dir == "" else dest_dir
        ans_file_path = join(dest_dir , dirname(dir_path).strip(".txt") + ".ans")
        dest_file = open(ans_file_path ,'w')
    
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
       
    if save:
        dest_dir = dirname(file_path) if dest_dir == "" else dest_dir
        ans_file_path = join(dest_dir , basename(file_path).strip(".txt") + ".ans")
        dest_file = open(ans_file_path ,'w')
    
    for i, sentence in enumerate(textHandler.get_sentences_from_file(file_path)):
        if(limit != -1 and i >= limit): break
        answer = get_answer_func(sentence)
        if(answer != ""):
            answers.append(answer)
            sentences.append(sentence)            
            if save:
                dest_file.write(sentence + "\n")
                dest_file.write(answer + "\n")
    
    if save:
        dest_file.close()

    return sentences, answers    

def get_sentences_and_answers_from_existing_file(file_path, limit = -1):
    sentences = []
    answers = []
    limit *= 2
    
    for i, sentence in enumerate(textHandler.get_sentences_from_file(file_path)):
        if(limit != -2 and i >= limit): break
        if(i%2 == 0):
            sentences.append(sentence + " ;")
        else:
            answers.append(sentence + " ;")

    return sentences, answers    

def get_sentences_and_answers(src_path, get_answer_func = None, save = 1 ,dest_dir = "", limit = -1):
    if not isdir(src_path) and not isfile(src_path):
        return None, None
    if dest_dir == "":
        dest_dir = src_path

    if isfile(src_path):
        return get_sentences_and_answers_by_file_path(src_path,get_answer_func,save,dest_dir, limit)
    elif isdir(src_path):
        for filename in listdir(dest_dir): 
            if filename.endswith(".ans"):
                ans_file_path = join(src_path,filename)
                return get_sentences_and_answers_from_existing_file(ans_file_path, limit)
        return get_sentences_and_answers_by_dir_path(src_path,get_answer_func,save,dest_dir, limit)
    else:
        print("Error!")
        return None, None

