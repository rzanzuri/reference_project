import numpy as np
import Python_lib.textHandler as textHandler
from os import listdir
from os.path import basename , dirname, exists, join , isfile, isdir, split
from Python_lib.NER import get_ner_sentence
from nltk.tokenize import regexp_tokenize

# def get_data_set(sentences, model, vec_len, test_size):
#     test_set_size = int(len(sentences) * test_size)
#     data_set_size = len(sentences) - test_set_size
#     max_sen_len = textHandler.get_len_of_longest_sentence(sentences)

#     data = np.zeros((data_set_size, max_sen_len, vec_len))
#     test = np.zeros((test_set_size, max_sen_len, vec_len))

#     for i , sent in  enumerate(sentences):
#         for j ,word in enumerate(regexp_tokenize(sent, pattern=r'\w+|\$[\d\.]+|\S+')):
#             if word in model.wv.vocab and i < data_set_size:
#                 data[i,j] = model.wv[word]
#             elif word in model.wv.vocab and i >= data_set_size:
#                 test[i - data_set_size,j] = model.wv[word]
#             elif i < data_set_size:
#                 data[i,j] = word
#             elif i >= data_set_size:
#                 test[i - data_set_size,j] = word                

#     return data, test

def get_data_set(sentences, model, test_size):
    test_set_size = int(len(sentences) * test_size)
    data_set_size = len(sentences) - test_set_size
    max_sen_len = textHandler.get_len_of_longest_sentence(sentences)

    if is_int(sentences[0]):
        data, test = get_binary_data_and_test(sentences, model, data_set_size, max_sen_len)
    else:
        data, test = get_strings_data_and_test(sentences, model, data_set_size, test_set_size, max_sen_len)               

    return data, test    

def get_binary_data_and_test(sentences, model, data_set_size, max_sen_len):
    data = np.zeros([data_set_size], dtype=np.int32)
    test = np.zeros([len(sentences) - data_set_size], dtype=np.int32)

    for i, sent in enumerate(sentences):
        if i < data_set_size:
            data[i] = int(sent)
        else:
            test[i - data_set_size] = int(sent)

    return data, test 

def get_strings_data_and_test(sentences, model, data_set_size, test_set_size, max_sen_len):
    data = np.zeros((data_set_size, max_sen_len), dtype=np.int32)
    test = np.zeros((test_set_size, max_sen_len), dtype=np.int32)

    for i, sent in enumerate(sentences):
        for j ,word in enumerate(regexp_tokenize(sent, pattern=r'\w+|\$[\d\.]+|\S+')):
            if word in model.wv.vocab and i < data_set_size:
                data[i,j] = model.wv.vocab[word].index
            elif word in model.wv.vocab and i >= data_set_size:
                test[i - data_set_size,j] = model.wv.vocab[word].index
            elif i < data_set_size:
                data[i,j] = 0
            elif i >= data_set_size:
                test[i - data_set_size,j] = 0                 

    return data, test 


def get_sentences_and_answers_by_dir_path(dir_path, get_answer_func, save = 1 ,dest_dir = "", max_len_sent = -1, limit = -1):
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
        dir_path = dir_path[:-1] if dir_path[-1] == '/' else dir_path
        ans_file_path = join(dest_dir , split(dir_path)[1] + ".ans")
        dest_file = open(ans_file_path ,'w', encoding='utf-8')
    
    for i, sentence in enumerate(textHandler.get_sentences_from_dir(dir_path, max_len_sent)):
        if(limit != -1 and i >= limit): break
        answer = get_answer_func(sentence)
        if(answer != ""):
            answers.append(answer)
            sentences.append(sentence)            
            if(save):
                try:
                    dest_file.write(sentence + "\n")
                    dest_file.write(str(answer) + "\n")
                except:
                    print(sentence)
                    print("\n")
                    print(answer)
                    print("\n")
                        

    if(save):
        dest_file.close()
    return sentences, answers

def get_sentences_and_answers_by_file_path(file_path, get_answer_func, save = 1 ,dest_dir = "", max_len_sent = -1, limit = -1):
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
        dest_file = open(ans_file_path ,'w', encoding='utf-8')
    
    for i, sentence in enumerate(textHandler.get_sentences_from_file(file_path, max_len_sent)):
        if(limit != -1 and i >= limit): break
        answer = get_answer_func(sentence)
        if(answer != ""):
            answers.append(answer)
            sentences.append(sentence)            
            if save:
                try:
                    dest_file.write(sentence + "\n")
                    dest_file.write(answer + "\n")
                except:
                    print(sentence + "\n")
                    print(answer + "\n")

    if save:
        dest_file.close()

    return sentences, answers    

def get_sentences_and_answers_from_existing_file(file_path, limit = -1):
    sentences = []
    answers = []
    limit *= 2
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, sentence in enumerate(f.readlines()):
            if(limit != -2 and i >= limit): break
            if(i%2 == 0):
                sentences.append(sentence)
            else:
                answers.append(sentence)

    return sentences, answers    

def get_sentences_and_answers(src_path, get_answer_func = None, save = 1 , dest_dir = "", max_len_sent = -1, limit = -1):
    if not isdir(src_path) and not isfile(src_path):
        return None, None
    if dest_dir == "":
        dest_dir = src_path

    if isfile(src_path):
        return get_sentences_and_answers_by_file_path(src_path, get_answer_func, save, dest_dir, max_len_sent, limit)
    elif isdir(src_path):
        for filename in listdir(dest_dir): 
            if filename.endswith(".ans"):
                ans_file_path = join(src_path,filename)
                return get_sentences_and_answers_from_existing_file(ans_file_path, limit)
        return get_sentences_and_answers_by_dir_path(src_path, get_answer_func, save, dest_dir, max_len_sent, limit)
    else:
        print("Error! get_sentences_and_answers")
        return None, None

def is_int(str):
    try: 
        int(str)
        return True
    except ValueError:
        return False