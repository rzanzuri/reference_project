import numpy as np
import tensorflow as tf
from os import listdir
from os.path import exists, join , isfile, isdir
import datetime
import random

# def get_train_and_test_sets(sentences, vec_model, test_size):
#     print("Start get_train_and_test_sets function", datetime.datetime.now())
#     test_set_size = int(len(sentences) * test_size)
#     train_set_size = len(sentences) - test_set_size
#     max_sen_len = max_sentence_length(sentences)

#     train_set = create_data_sets(sentences, vec_model)               
  
#     print("End get_train_and_test_sets function", datetime.datetime.now())
#     return train_set, test_set     

# def get_binary_data_and_test(sentences, model, data_set_size, max_sen_len):
#     data = np.zeros([data_set_size], dtype=np.int32)
#     test = np.zeros([len(sentences) - data_set_size], dtype=np.int32)

#     for i, sent in enumerate(sentences):
#         if i < data_set_size:
#             data[i] = int(sent)
#         else:
#             test[i - data_set_size] = int(sent)

#     return data, test 

def get_data_sets(sentences, vec_model, non_exists_idx_val = 0, pad_idx_val = 0):
    print("Start get_data_sets function", datetime.datetime.now())
    data_set = []
    #creates train data set
    for i, sentence in enumerate(sentences):
        data_set.append(list())
        for word in sentence.split():
            if word in vec_model.wv.vocab:
                data_set[i].append(vec_model.wv.vocab[word].index)
            else: 
                data_set[i].append(non_exists_idx_val); #default index for non-exsits in vec_model voacb
    data_set = tf.keras.preprocessing.sequence.pad_sequences(data_set, padding='post', value=pad_idx_val)

    print("End get_data_sets function", datetime.datetime.now())
    return data_set 

def get_binary_data_set(data_set):
    print("Start get_data_sets function", datetime.datetime.now())
    data_set = np.array(data_set)
    print("End get_data_sets function", datetime.datetime.now())
    return data_set

def sentence_to_indexes(sentence, vec_model, max_length_inp):
  indexes = []
  for word in sentence.split(' '):
    if word in vec_model.wv.vocab:
      indexes.append(vec_model.wv.vocab[word].index)
    else:
      indexes.append(0)

  return tf.keras.preprocessing.sequence.pad_sequences([indexes], maxlen=max_length_inp, padding='post', value=vec_model.wv.vocab['<end>'].index)

def indexes_to_sentence(indexes, vec_model):
  sentence = ''
  for index in indexes:
    if index >= 0:
      sentence += vec_model.wv.index2word[index] + ' '
    else:
      sentence+= 'NONE '
  return sentence

def create_sentences_and_answers(file_path, get_ans_func, start_tag = "", end_tag = "", max_len_sent = -1, limit = -1, do_shuffle = 0):
    ans_file = file_path.strip(".txt") + ".ans"
    
    with open(file_path, 'r', encoding='utf-8') as rf:
        with open(ans_file, 'w', encoding='utf-8') as wf:
            for sentence in rf.readlines():
                answer = get_ans_func(sentence)
                wf.write(sentence.strip().rstrip() + "\n")
                wf.write(answer.strip().rstrip() + "\n")

    return read_sentences_and_answers(ans_file, start_tag, end_tag, max_len_sent, limit, do_shuffle)        

def read_sentences_and_answers(file_path, start_tag = "", end_tag = "", max_len_sent = -1, limit = -1, do_shuffle = 0):

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    sentences = []
    answers = []
    start_tag = start_tag if start_tag == "" else start_tag + " "
    end_tag = end_tag if end_tag == "" else " " + end_tag
    
    sent_ans_pairs = list(zip(lines[0::2],lines[1::2])) # creats piars of even (sentences) ans odd (answers) line
    if do_shuffle: 
        random.shuffle(sent_ans_pairs)

    for sentence, answer in sent_ans_pairs:
        sentence = sentence.strip().rstrip()
        answer = answer.strip().rstrip()
        if max_len_sent == -1 or len(sentence.split()) <= max_len_sent:
            sentence = start_tag + sentence + end_tag
            answer = start_tag + answer + end_tag
            if is_int(answer): answer = int(answer)
            sentences.append(sentence)
            answers.append(answer)

    if limit > 0:
        sentences = sentences[:limit]
        answers   = answers[:limit]

    return sentences, answers    

def get_sentences_and_answers(src_path, start_tag = "", end_tag = "", max_len_sent = -1, limit = -1, do_shuffle = 0, get_answer_func = None):
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
        sentences, answers = read_sentences_and_answers(src_file, start_tag, end_tag, max_len_sent, limit, do_shuffle)
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
