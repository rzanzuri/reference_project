import gensim
import datetime
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils


sentences = []
with open('./sentencesV1.txt') as f:
    sentences = f.read().splitlines()

answer = []
with open('./answerV1.txt') as f:
    answer = f.read().splitlines()

model = gensim.models.Word2Vec.load("./mymodel")
# model = gensim.models.KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary=True)  

flatten = []
dataX = []
dataY = []
lineX=[]
counter = 9

for line in sentences:
    flatten.append(line.split(" "))

for line in flatten:
    for word in line:
        if word in model.wv.vocab:
            counter = counter - 1
            lineX.append(model[word])
        else:
            lineX.append(np.zeros(300))

    for i in range(counter):
        lineX.append(np.zeros(300))
    
    counter=9
    dataX.append(lineX)
    lineX=[]

for word in answer:
    if word in model.wv.vocab:
        dataY.append(model[word])
    else:
         dataY.append(np.zeros(300))   
    
dataset = np.reshape(dataX,(29,9,300))
dataAns = np.reshape(dataY,(29,300))

#attention

model = Sequential()
model.add(LSTM(256, input_shape=(9,300)))
model.add(Dropout(0.2))
model.add(Dense(300, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')
filepath="weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=0, save_best_only=True, mode='min')
callbacks_list = [checkpoint]
# fit the model
model.fit(dataset, dataAns, epochs=10, batch_size=128, callbacks=callbacks_list)
model.save("./test_traning")