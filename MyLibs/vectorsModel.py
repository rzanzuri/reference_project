import gensim
import random
import datetime
from gensim.models import word2vec
from gensim.models.word2vec import LineSentence
import logging
# import MyLibs.textHandler as textHandler
from os.path import basename , isdir , isfile , join , exists, dirname
from os import listdir, mkdir
import numpy as np
import copy

def utf8len(s):
    return len(s.encode('utf-8'))

def get_shuffled_text(pars):
    text = ""
    random.shuffle(pars)
    for p in pars:
        text += p
    return text

def temp_for_dror(dest_dir, win_size, iters = 10, min_count = 10, size = 300, workers = 4):
    pars = []
    par = ""
    par_size = 0
    text_size = 0
    with open("./Data/clean_shut/clean_shut.txt",'r', encoding='utf-8') as f:
        text = f.read()
        lines = text.splitlines()
        text_size = utf8len(text)
        max_par_size = text_size / 7
        num_of_lines = len(lines) - 1
        
        print("Start split the text at: " + str(datetime.datetime.now()))

        for i, line in enumerate(lines):
            par += line
            par_size += utf8len(line)
            if par_size >= max_par_size or i == num_of_lines:
                pars.append(par)
                par = ""
                par_size = 0
        
        print("End split the text at: " + str(datetime.datetime.now()))

        print("Start write shuffled text at: " + str(datetime.datetime.now()))

        with open("./Data/shuffled_clean_shut/shuffled_clean_shut.txt",'w',encoding='utf-8') as f2:
            f2.write(text)
            f2.write(get_shuffled_text(pars))
            f2.write(get_shuffled_text(pars))
        print("End write shuffled text at: " + str(datetime.datetime.now()))

        return  create_vectors_model("./Data/shuffled_clean_shut/shuffled_clean_shut.txt",dest_dir,win_size, iters,min_count, size,workers)      

def create_vectors_model(src_path, src_file, win_size, iters = 10, min_count = 10, size = 300, workers = 4, non_exists_word = 'NONE', special_tags = []):   
    print("Start create model vectors at: " , datetime.datetime.now())
    
    sentences = LineSentence(join(src_path, src_file))
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    vec_model = word2vec.Word2Vec(sentences, window = win_size, iter = iters, min_count = min_count, size = size, workers = workers)

    model_path = get_vec_model_path(src_path, src_file)
    vec_model.save(model_path)

    vec_model = create_special_tags_vectors(non_exists_word, special_tags, vec_model)

    print("End create model vectors at: " , datetime.datetime.now())

    return vec_model
 
def load_vectors_model(model_path, non_exists_word = 'NONE', special_tags = []):
    vec_model = gensim.models.Word2Vec.load(model_path)
    vec_model = create_special_tags_vectors(non_exists_word, special_tags, vec_model)
    return vec_model

def get_model_vectors(src_path, win_size = 10 ,iters = 10, min_count = 10, size = 300, workers = 4, non_exists_word = 'NONE', special_tags = []):
    print("Start get_model_vectors function", datetime.datetime.now())
    src_file = None
    if isdir(src_path):
        src_file = get_first_txt_file(src_path)
    elif isfile(src_path) and src_path.endswith(".txt"):
        src_file = basename(src_path)
        src_path = dirname(src_path)
    
    if src_file == None:
        print("ERROR: get_model_vectors - Text file not found")

    vec_model_path = get_vec_model_path(src_path, src_file)
   
    if exists(vec_model_path):
        vec_model = load_vectors_model(vec_model_path, non_exists_word, special_tags)
    else:
        vec_model = create_vectors_model(src_path, src_file, win_size, iters , min_count, size, workers, non_exists_word, special_tags)
    
    print("End get_model_vectors function", datetime.datetime.now())
    return vec_model


def get_index_vectors_matrix(vec_model):
    return vec_model.wv.syn0, vec_model.wv.syn0.shape[0], vec_model.wv.syn0.shape[1]
def get_vec_model_dir(src_path):
    full_path = join(src_path, "VecModels/")
    if not exists(full_path):
        mkdir(full_path)
    return full_path

def get_first_txt_file(src_path):
    for filename in listdir(src_path): 
        if filename.endswith(".txt"):
            return filename
    return None

def get_vec_model_path(src_path, src_file):
    vec_model_dir_path = get_vec_model_dir(src_path)
    vec_model_file_name = src_file.strip(".txt")  + ".vm"
    return vec_model_dir_path + vec_model_file_name

def create_special_tags_vectors(non_exists_word, special_tags ,vec_model):

    if non_exists_word in  vec_model.wv.vocab:
        print("ERROR: create_special_tags_vectors non exists word already exsits in vec_model")
    
    special_tags.append(non_exists_word)

    for tag in special_tags:
        if tag not in vec_model.wv.vocab:
            sent = (tag + ' ') * vec_model.min_count 
            vec_model.build_vocab([sent.split()], update=True)
        else:
            print("WARNING: create_special_tags_vectors - tag {0} already exists in vec_model vocab".format(tag))
    
    replace_idx_0_word(non_exists_word, vec_model)

    return vec_model

    
def replace_idx_0_word(non_exists_word, vec_model):
    idx_0_idx = 0
    # idx_0_word = vec_model.wv.index2word[idx_0_idx]
    # idx_0_vec = copy.deepcopy(vec_model.wv[idx_0_word])

    # non_exists_idx = vec_model.wv.vocab[non_exists_word].index
    # non_exists_word = non_exists_word
    # non_exists_vec = np.zeros((vec_model.wv.syn0.shape[1]))

    # vec_model.wv.vocab[idx_0_idx] = non_exists_idx
    # vec_model.wv.index2word[idx_0_idx] = non_exists_word
    # vec_model.wv.syn0[idx_0_idx] = non_exists_vec

    # vec_model.wv.vocab[idx_0_word] = non_exists_idx
    # vec_model.wv.index2word[idx_0_idx] = non_exists_word
    # vec_model.wv.syn0[idx_0_idx] = non_exists_vec

    # vec_model.wv.vocab[non_exists_word] = non_exists_idx
    # vec_model.wv.index2word[non_exists_idx] = idx_0_word
    # vec_model.wv.syn0[non_exists_idx] = idx_0_vec

    # vec_model.wv.init_sims()