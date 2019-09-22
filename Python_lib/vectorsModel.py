import gensim
import random
import datetime
from gensim.models import word2vec
from gensim.models.word2vec import LineSentence
import logging
# import Python_lib.textHandler as textHandler
from os.path import basename , isdir , isfile , join , exists
from os import listdir

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

        return  get_model_vectors_by_file_path("./Data/shuffled_clean_shut/shuffled_clean_shut.txt",dest_dir,win_size, iters,min_count, size,workers)      
  

def get_model_vectors_by_file_path(file_path, dest_dir, win_size, iters = 10, min_count = 10, size = 300, workers = 4):   
    # sentences = textHandler.get_sentences(file_path)
    print("Start create model vectors at: " + str(datetime.datetime.now()))

    sentences = LineSentence(file_path)
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    model = word2vec.Word2Vec(sentences, window = win_size, iter = iters, min_count = min_count, size = size, workers = workers)
    file_name = basename(file_path).strip(".txt") + ".vm"
    model_path = join(dest_dir ,file_name )
    model.save(model_path)
    model.similar_by_vector()

    print("End create model vectors at: " + str(datetime.datetime.now()))

    return model
 
def get_existing_vectors_model(model_path):
    return gensim.models.Word2Vec.load(model_path)

def get_model_vectors(src_path, dest_dir, win_size ,iters = 10, min_count = 10, size = 300, workers = 4):
    if not isdir(src_path) or not isdir(dest_dir):
        return None
    for filename in listdir(src_path): 
        if filename.endswith(".txt"):
            model_name = join(dest_dir,filename.strip(".txt")) + ".vm"
            if exists(model_name):
                return get_existing_vectors_model(model_name)
            else:
                file_path = join(src_path, filename)
                return get_model_vectors_by_file_path(file_path, dest_dir,win_size, iters , min_count, size, workers)

    print("ERROR: Text file not found")
    return None  