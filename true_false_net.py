from keras.models import Sequential
from keras import layers
from keras.layers import Embedding, LSTM, Dense, Activation, Flatten, Bidirectional
from keras.models import model_from_json
import MyLibs.dataSets as dataSets
import MyLibs.vectorsModel as vectorsModel
import MyLibs.named_entity_recognotion as NER
import MyLibs.textHandler as textHandler
import numpy as np
import multiprocessing
import logging
import datetime
import copy

import gensim

from gensim.test.utils import get_tmpfile
iters = 50
min_count = 20
vec_size = 300 
win_size = 10
workers = multiprocessing.cpu_count() 
epochs = 50
limit = 1000
test_size = 0.25
max_len_sent = 100


start = datetime.datetime.now()
print("start:", start)

#model setup
vec_model_root_path = "./VecModels/"
# curpus_path = "./Data/Wiki_en/"
# curpus_path = "./Data/sentiment_analysis/"
# curpus_path = "./Data/shuffled_clean_shut/"
curpus_path = "./Data/RabannyText/"
# curpus_path = "./Data/HebrewText/"
# curpus_path = "./Data/‏‏HebrewTextForNER/"
# curpus_path = "./Data/HebrewTextPart/"
# curpus_path = "./Data/sentiment labelled sentences/"

print("curpus_path:", curpus_path)
# curpus_path = "./Data/EnglishTextForNER/"

# vec_model_curpus_path =  "./Data/sentiment_analysis/"
# vec_model = vectorsModel.temp_for_dror(vec_model_root_path, win_size, iters, min_count, vec_size, workers)
# print(vec_model.wv['שאלה'])

# vec_model = gensim.models.KeyedVectors.load_word2vec_format('./VecModels/GoogleNews-vectors-negative300.bin', binary=True)  


vec_model = vectorsModel.get_model_vectors(curpus_path, vec_model_root_path, win_size, iters, min_count, vec_size, workers)


# x =  copy.deepcopy(vec_model.wv)
# if "<start>" not in x.vocab:
#     print("Not In X")

# sent = 'שאלה ' * 30
# vec_model.build_vocab([sent.split()], update=True)

# if "<start>" in vec_model.wv.vocab:
#     print("In Vocab")

# if "<start>" not in x.vocab:
#     print("Not In X")
# i = 0
# for w in x.vocab:
#     s1 = vec_model.wv[w]
#     s2 = x[w]
#     if not np.allclose(s1,s2):
#         i+=1
#         print("Changed",i)
exit()
sentences, answers = dataSets.get_sentences_and_answers(curpus_path, NER.is_ner_exsits, max_len_sent = max_len_sent, limit= limit)

X_train, X_test = dataSets.get_train_and_test_sets(sentences,vec_model, test_size)
Y_train, Y_test = dataSets.get_train_and_test_sets(answers, vec_model, test_size)

pretrained_weights = vec_model.wv.syn0
vocab_size, emdedding_size = pretrained_weights.shape

# print(pretrained_weights)
# print(pretrained_weights[0])
# print(pretrained_weights[0][0])

print('Result embedding shape:', pretrained_weights.shape)
print('train_x shape:', X_train.shape)
print('train_y shape:', Y_train.shape)

maxlen = X_train.shape[1]

model = Sequential()
model.add(layers.Embedding(vocab_size, emdedding_size, 
                           weights=[pretrained_weights], 
                           input_length=maxlen, 
                           trainable=True))
model.add(layers.GlobalMaxPool1D())
# model.add(LSTM(units=emdedding_size))
# model.add(Bidirectional(LSTM(units=emdedding_size)))

model.add(layers.Dense(10, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.summary()

# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

history = model.fit(X_train, Y_train,
                    epochs=epochs,
                    verbose=False,
                    validation_data=(X_test, Y_test))



loss, accuracy = model.evaluate(X_train, Y_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(X_test, Y_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))
model.save("./RabbnyText_regular.md")

finish = datetime.datetime.now()
print("end:", finish)
print("total:", finish - start)
