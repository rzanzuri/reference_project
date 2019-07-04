from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding, Flatten, TimeDistributed, Dropout, LSTMCell, RNN
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.utils import tf_utils
from tensorflow.keras import backend as K
import tensorflow as tf
import Python_lib.dataSets as dataSets
import Python_lib.vectorsModel as vectorsModel
import Python_lib.named_entity_recognotion as NER
import Python_lib.textHandler as textHandler
import multiprocessing
import logging
import datetime

import gensim
# This enables the Jupyter backend on some matplotlib installations.
# %matplotlib notebook
import matplotlib.pyplot as plt
# Turn off interactive plots. iplt doesn't work well with Jupyter.
plt.ioff()

import unicodedata
import re
import numpy as np
import os
import time
import shutil

iters = 6
min_count = 60
vec_size = 300 
win_size = 10
workers = multiprocessing.cpu_count() 
epochs = 1
limit = 1000
test_size = 0.2
max_len_sent = 100
BATCH_SIZE = 10

start = datetime.datetime.now()
print("start:", start)

#model setup

curpus_path = "./Data/RabannyText/"
vec_model_root_path = "./VecModels/"

print("curpus_path:", curpus_path)

vec_model = vectorsModel.get_model_vectors(curpus_path, vec_model_root_path, win_size, iters, min_count, vec_size, workers)

sent = '<start> ' * 30
vec_model.build_vocab([sent.split()], update=True)

sent = '<end> ' * 30
vec_model.build_vocab([sent.split()], update=True)

sent = '<start_ref> ' * 30
vec_model.build_vocab([sent.split()], update=True)

sent = '<end_ref> ' * 30
vec_model.build_vocab([sent.split()], update=True)

sentences, answers = dataSets.get_sentences_and_answers(curpus_path, NER.is_ner_exsits, max_len_sent = max_len_sent, limit= limit)

X_train, _ = dataSets.get_data_set(sentences,vec_model, 0)
Y_train, _ = dataSets.get_data_set(answers, vec_model, 0)
input_len = X_train.shape[1]
output_len  = Y_train.shape[1]

target_data = [[Y_train[n][i+1] for i in range(len(Y_train[n])-1)] for n in range(len(Y_train))]
target_data = tf.keras.preprocessing.sequence.pad_sequences(target_data, maxlen=output_len, padding="post")
target_data = target_data.reshape((target_data.shape[0], target_data.shape[1], 1))

pretrained_weights = vec_model.wv.syn0
vocab_size, emdedding_size = pretrained_weights.shape

print('Result embedding shape:', pretrained_weights.shape)
print('train_x shape:', X_train.shape)
print('train_y shape:', Y_train.shape)
print('target_data shape:', target_data.shape)


# Create the Encoder layers first.
encoder_inputs = Input(shape=(input_len,))
encoder_emb = Embedding(vocab_size, emdedding_size, weights=[pretrained_weights], input_length=input_len, trainable=True)
encoder_lstm = LSTM(units=emdedding_size, return_sequences=True, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(encoder_emb(encoder_inputs))
encoder_states = [state_h, state_c]

# Now create the Decoder layers.
decoder_inputs = Input(shape=(None,))
decoder_emb = Embedding(vocab_size, emdedding_size, weights=[pretrained_weights], input_length=output_len, trainable=True)
decoder_lstm = LSTM(units=emdedding_size, return_sequences=True, return_state=True)
decoder_lstm_out, _, _ = decoder_lstm(decoder_emb(decoder_inputs), initial_state=encoder_states)
# Two dense layers added to this model to improve inference capabilities.
decoder_d1 = Dense(emdedding_size, activation="relu")
decoder_d2 = Dense(vocab_size, activation="softmax")
# Drop-out is added in the dense layers to help mitigate overfitting in this part of the model. Astute developers
# may want to add the same mechanism inside the LSTMs.
decoder_out = decoder_d2(Dropout(rate=.4)(decoder_d1(Dropout(rate=.4)(decoder_lstm_out))))

# Finally, create a training model which combines the encoder and the decoder.
# Note that this model has three inputs:
#  encoder_inputs=[batch,encoded_words] from input language (English)
#  decoder_inputs=[batch,encoded_words] from output language (Spanish). This is the "teacher tensor".
#  decoder_out=[batch,encoded_words] from output language (Spanish). This is the "target tensor".
model = Model([encoder_inputs, decoder_inputs], decoder_out)
# We'll use sparse_categorical_crossentropy so we don't have to expand decoder_out into a massive one-hot array.
#  Adam is used because it's, well, the best.
model.compile(optimizer=tf.keras.optimizers.Adam(), loss="sparse_categorical_crossentropy", metrics=['sparse_categorical_accuracy'])
# Note, we use 20% of our data for validation.
history = model.fit([X_train, Y_train], target_data,
                 batch_size=BATCH_SIZE,
                 epochs=epochs,
                 validation_split=test_size)

model.save("./new_net.md")
# Plot the results of the training.
plt.plot(history.history['sparse_categorical_accuracy'], label="Training loss")
plt.plot(history.history['val_sparse_categorical_accuracy'], label="Validation loss")
plt.show()