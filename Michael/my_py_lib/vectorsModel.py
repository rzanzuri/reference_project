import gensim
from gensim.models import word2vec
import logging
import my_py_lib.textHandler as textHandler
from os.path import basename , isdir , isfile
  

def get_model_vectors_by_file_path(file_path, dest_dir, iters = 10, min_count = 10, size = 300, workers = 4):   
    # sentences = textHandler.get_sentences(file_path)
    sentences = word2vec.Text8Corpus(file_path)
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    model = word2vec.Word2Vec(sentences, iter = iters, min_count=min_count, size=size, workers=workers)
    file_name = basename(file_path)
    model.save(dest_dir + file_name + ".vecmodel")
    return model

# def get_model_vectors_by_dir_path(dir_path, dest_dir, iters = 10, min_count = 10, size = 300, workers = 4):
#     if(not dest_dir.endswith("\\")):
#         dest_dir = dest_dir + "\\"
#     sentences = textHandler.get_sentences_from_dir(dir_path)
#     logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
#     model = word2vec.Word2Vec(sentences, iter = iters, min_count=min_count, size=size, workers=workers)
#     file_name = basename(dir_path)
#     model.save(dest_dir + file_name + ".vecmodel")
#     return model
 
def get_existing_vectors_model(model_path):
    return gensim.models.Word2Vec.load(model_path)

def get_model_vectors(path, dest_dir, iters = 10, min_count = 10, size = 300, workers = 4):
    if(isdir(path)):
        return None
        # return get_model_vectors_by_dir_path(path, dest_dir, iters = 10, min_count = 10, size = 300, workers = 4)
    elif isfile(path) and not path.endswith(".vecmodel"):
        return get_model_vectors_by_file_path(path, dest_dir, iters = 10, min_count = 10, size = 300, workers = 4)
    elif isfile(path) and path.endswith(".vecmodel"):
        return get_existing_vectors_model(path)
    else:
        print("ERROR: get_model_vectors")
        return None


