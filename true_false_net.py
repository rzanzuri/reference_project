from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, Bidirectional, GlobalMaxPool1D
from sklearn.model_selection import train_test_split
from os.path import join

import MyLibs.dataSets as dataSets
import MyLibs.vectorsModel as vectorsModel
import MyLibs.named_entity_recognotion as NER
import multiprocessing
import datetime
start = datetime.datetime.now()
print("start:", start)

corpus_path, ans_kind = "./Data/RabannyText/", "RabannyText_true_false.ans"

workers = multiprocessing.cpu_count() 
epochs = 50
num_examples =  -1
max_len_sent = -1
test_size = 0.25
batch_size = 1
do_shuffle = 0

print("\n\n-----------------------------------------------------")
print("Setup:")
print("corpus_path:", corpus_path)
print("workers:", workers)
print("epochs:", epochs)
print("num_examples:", num_examples)
print("test_size:", test_size)
print("batch_size:", batch_size)
print("do_shuffle:", do_shuffle)
print("-----------------------------------------------------\n\n")

#gets/creates gensim vectors model of corpus
vec_model = vectorsModel.get_model_vectors(corpus_path, iters= 50, min_count = 40, workers = workers)
pretrained_weights, vocab_size, emdedding_size = vectorsModel.get_index_vectors_matrix(vec_model)
sentences, answers = dataSets.get_sentences_and_answers(join(corpus_path,ans_kind), max_len_sent = max_len_sent, limit = num_examples, do_shuffle = do_shuffle)

sentences_train, sentences_test, answers_train, answers_test = train_test_split(sentences, answers, test_size = test_size)

#transform sentences and answers to idexes (in vocab) vectors, and split to trian and test sets
X_train = dataSets.get_data_sets(sentences_train, vec_model)
Y_train = dataSets.get_binary_data_set(answers_train)

X_test = sentences_test
Y_test = answers_test

print('Result embedding shape:', pretrained_weights.shape)
print('X_train, X_test length:',  len(X_train),  len(X_test))
print('Y_train, Y_test length:',  len(Y_train),  len(Y_test))

maxlen = X_train.shape[1]

model = Sequential()
model.add(Embedding(vocab_size, emdedding_size, 
                           weights=[pretrained_weights], 
                           input_length=maxlen, 
                           trainable=True))
# model.add(GlobalMaxPool1D())
model.add(LSTM(units=emdedding_size))
# model.add(Bidirectional(LSTM(units=emdedding_size)))

model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.summary()

history = model.fit(X_train, Y_train,
                    epochs=epochs,
                    verbose=False,
                    validation_data=(X_test, Y_test))

loss, accuracy = model.evaluate(X_train, Y_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(X_test, Y_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))
model.save(join(corpus_path, "/RabbnyText_regular.md"))

finish = datetime.datetime.now()
print("end:", finish)
print("total:", finish - start)
