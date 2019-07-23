import numpy as np
from os import listdir
from os.path import exists, join , isfile, isdir
import datetime
import random

def get_train_and_test_sets(sentences, vec_model, test_size):
    print("Start get_train_and_test_sets function", datetime.datetime.now())
    test_set_size = int(len(sentences) * test_size)
    train_set_size = len(sentences) - test_set_size
    max_sen_len = max_sentence_length(sentences)

    train_set, test_set = create_data_sets(sentences, vec_model, train_set_size, test_set_size, max_sen_len)               
  
    print("End get_train_and_test_sets function", datetime.datetime.now())
    return train_set, test_set     

# def get_binary_data_and_test(sentences, model, data_set_size, max_sen_len):
#     data = np.zeros([data_set_size], dtype=np.int32)
#     test = np.zeros([len(sentences) - data_set_size], dtype=np.int32)

#     for i, sent in enumerate(sentences):
#         if i < data_set_size:
#             data[i] = int(sent)
#         else:
#             test[i - data_set_size] = int(sent)

#     return data, test 

def create_data_sets(sentences, vec_model, train_set_size, test_set_size, max_sen_len):
    train_set = np.zeros((train_set_size, max_sen_len), dtype=np.int32)
    test_set  = np.zeros((test_set_size, max_sen_len) , dtype=np.int32)

    #creates train data set
    for i, sentence in enumerate(sentences[:train_set_size]):
        for j ,word in enumerate(sentence.split()):
            if word in vec_model.wv.vocab:
                train_set[i,j] = vec_model.wv.vocab[word].index
            else: 
                train_set[i,j] = 0 #default index 0 for non-exsits in vec_model voacb
    
    for i, sentence in enumerate(sentences[train_set_size:]):
        for j ,word in enumerate(sentence.split()):
            if word in vec_model.wv.vocab:
                test_set[i,j] = vec_model.wv.vocab[word].index
            else:
                test_set[i,j] = 0 #default index 0 for non-exsits in vec_model voacb             

    return train_set, test_set 


def create_sentences_and_answers(file_path, get_ans_func, start_tag = "", end_tag = "", max_len_sent = -1, limit = -1):
    ans_file = file_path.strip(".txt") + ".ans"
    
    with open(file_path, 'r', encoding='utf-8') as rf:
        with open(ans_file, 'w', encoding='utf-8') as wf:
            for sentence in rf.readlines():
                answer = get_ans_func(sentence)
                wf.write(sentence.strip().rstrip() + "\n")
                wf.write(answer.strip().rstrip() + "\n")

    return read_sentences_and_answers(ans_file, start_tag, end_tag, max_len_sent, limit)        

def read_sentences_and_answers(file_path, start_tag = "", end_tag = "", max_len_sent = -1, limit = -1):

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    sentences = []
    answers = []
    start_tag = start_tag if start_tag == "" else start_tag + " "
    end_tag = end_tag if end_tag == "" else " " + end_tag
    
    sent_ans_pairs = list(zip(lines[0::2],lines[1::2])) # creats piars of even (sentences) ans odd (answers) line
    random.shuffle(sent_ans_pairs)

    for sentence, answer in sent_ans_pairs:
        sentence = sentence.strip().rstrip()
        answer = answer.strip().rstrip()
        if max_len_sent == -1 or len(sentence.split()) <= max_len_sent:
            sentences.append(start_tag + sentence + end_tag)
            answers.append(start_tag + answer + end_tag)

    if limit > 0:
        sentences = sentences[:limit]
        answers   = answers[:limit]

    return sentences, answers    

def get_sentences_and_answers(src_path, start_tag = "", end_tag = "", max_len_sent = -1, limit = -1, get_answer_func = None):
    print("Start get_sentences_and_answers function", datetime.datetime.now())
    src_file = None
    if isdir(src_path):
        src_file = get_first_ans_file(src_path)
    elif isfile(src_path):
        src_file = src_path
    
    if src_file is None:
        print("ERROR: get_sentences_and_answers - file not found")
        return None, None

    if exists(src_file):
        sentences, answers = read_sentences_and_answers(src_file, start_tag, end_tag, max_len_sent, limit)
    else:
        sentences, answers = create_sentences_and_answers(src_file, get_answer_func, start_tag, end_tag, max_len_sent, limit)

    print("End get_sentences_and_answers function", datetime.datetime.now())
    return sentences, answers    


def is_int(str):
    try: 
        int(str)
        return True
    except ValueError:
        return False

def get_first_ans_file(src_path):
    for filename in listdir(src_path): 
        if filename.endswith(".ans"):
            return join(src_path, filename)
    return None

def max_sentence_length(sentences):
    return max(len(sentence.split()) for sentence in sentences)
