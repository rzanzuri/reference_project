import gensim
from gensim.models import word2vec
import logging
# import my_py_lib.textHandler as textHandler
from os.path import basename , isdir , isfile , join , exists
from os import listdir
  

def get_model_vectors_by_file_path(file_path, dest_dir, iters = 10, min_count = 10, size = 300, workers = 4):   
    # sentences = textHandler.get_sentences(file_path)
    sentences = word2vec.Text8Corpus(file_path)
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    model = word2vec.Word2Vec(sentences, iter = iters, min_count=min_count, size=size, workers=workers)
    file_name = basename(file_path).strip(".txt") + ".vm"
    model_path = join(dest_dir ,file_name )
    model.save(model_path)
    return model
 
def get_existing_vectors_model(model_path):
    return gensim.models.Word2Vec.load(model_path)

def get_model_vectors(src_path, dest_dir, iters = 10, min_count = 10, size = 300, workers = 4):
    if not isdir(src_path) or not isdir(dest_dir):
        return None
    for filename in listdir(src_path): 
        if filename.endswith(".txt"):
            model_name = join(dest_dir,filename.strip(".txt")) + ".vm"
            if exists(model_name):
                return get_existing_vectors_model(model_name)
            else:
                file_path = join(src_path, filename)
                return get_model_vectors_by_file_path(file_path, dest_dir, iters , min_count, size, workers)

    print("ERROR: Text file not found")
    return None  