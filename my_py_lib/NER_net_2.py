import gensim
import datetime
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM ,Bidirectional ,Dense , Dropout ,Input, Embedding
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
import nltk
import bz2
import os
from NER import get_ner_sentence
import dataSets
from vectorsModel import get_existing_vectors_model

test_mode = 0
if(test_mode):
    sentences, answers = short_test_text()
else:
    sentences, answers = en_wiki_test_text()    

model = get_existing_vectors_model("./mymodel")

vec_len = 300
num_of_sens = len(sentences)
sen_len = data_sets.get_len_of_longest_sentence(sentences)
ans_len = sen_len + 2

latent_dim = 256
epochs = 100
batch_size = num_of_sens

dataX = data_sets.get_data_set(sentences, model, vec_len)
dataY = data_sets.get_data_set(answers, model, vec_len)

encoder_input_data = np.reshape(dataX,(num_of_sens,sen_len,vec_len))
decoder_input_data  = np.reshape(dataY,(num_of_sens,ans_len,vec_len))
decoder_target_data  = np.reshape(dataY,(num_of_sens,ans_len ,vec_len))

input_words = get_text_words(sentences)
target_words = get_text_words(answers)
num_encoder_tokens = vec_len
num_decoder_tokens = vec_len

# Define an input sequence and process it.
encoder_inputs = Input(shape=(None,))
x = Embedding(num_encoder_tokens, latent_dim)(encoder_inputs)
x, state_h, state_c = LSTM(latent_dim,
                           return_state=True)(x)
encoder_states = [state_h, state_c]

# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(sen_len,vec_len))
x = Embedding(num_decoder_tokens, latent_dim)(decoder_inputs)
x = LSTM(latent_dim, return_sequences=True)(x, initial_state=encoder_states)
decoder_outputs = Dense(num_decoder_tokens, activation='softmax')(x)

# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

# Compile & run training
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
# Note that `decoder_target_data` needs to be one-hot encoded,
# rather than sequences of integers like `decoder_input_data`!
model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.2)