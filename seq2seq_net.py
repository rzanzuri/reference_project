from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding, Flatten, TimeDistributed, Dropout, LSTMCell, RNN
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.utils import tf_utils
from tensorflow.keras import backend as K
from Python_lib.LSTMWithAttention import LSTMWithAttention
import tensorflow as tf
import Python_lib.dataSets as dataSets
import Python_lib.vectorsModel as vectorsModel
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
BATCH_SIZE = 8

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

sentences, answers = dataSets.get_sentences_and_answers(curpus_path, max_len_sent = max_len_sent, limit= limit)

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


# attenc_inputs = Input(shape=(input_len,), name="attenc_inputs")
# attenc_emb = Embedding(vocab_size, emdedding_size, weights=[pretrained_weights], input_length=input_len, trainable=True)
# attenc_lstm = LSTM(units=emdedding_size, return_sequences=True, return_state=True)
# attenc_outputs, attstate_h, attstate_c = attenc_lstm(attenc_emb(attenc_inputs))
# attenc_states = [attstate_h, attstate_c]

# attdec_inputs = Input(shape=(None,))
# attdec_emb = Embedding(vocab_size, emdedding_size, weights=[pretrained_weights], input_length=output_len, trainable=True)
# attdec_lstm = LSTMWithAttention(units=emdedding_size, return_sequences=True, return_state=True)
# # Note that the only real difference here is that we are feeding attenc_outputs to the decoder now.
# # Nice and clean!
# attdec_lstm_out, _, _ = attdec_lstm(inputs=attdec_emb(attdec_inputs), 
#                                     constants=attenc_outputs, 
#                                     initial_state=attenc_states)
# attdec_d1 = Dense(emdedding_size, activation="relu")
# attdec_d2 = Dense(vocab_size, activation="softmax")
# attdec_out = attdec_d2(Dropout(rate=.4)(attdec_d1(Dropout(rate=.4)(attdec_lstm_out))))

# attmodel = Model([attenc_inputs, attdec_inputs], attdec_out)
# attmodel.compile(optimizer=tf.keras.optimizers.Adam(), loss="sparse_categorical_crossentropy", metrics=['sparse_categorical_accuracy'])

# atthist = attmodel.fit([X_train, Y_train], target_data,
#                  batch_size=BATCH_SIZE,
#                  epochs=epochs,
#                  validation_split=test_size)
# # Plot the results of the training.
# plt.plot(atthist.history['sparse_categorical_accuracy'], label="Training loss")
# plt.plot(atthist.history['val_sparse_categorical_accuracy'], label="Validation loss")
# plt.show()

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


model.save("./")
# Plot the results of the training.
# plt.plot(history.history['sparse_categorical_accuracy'], label="Training loss")
# plt.plot(history.history['val_sparse_categorical_accuracy'], label="Validation loss")
# plt.show()


# Create the encoder model from the tensors we previously declared.
encoder_model = Model(encoder_inputs, [encoder_outputs, state_h, state_c])

# Generate a new set of tensors for our new inference decoder. Note that we are using new tensors, 
# this does not preclude using the same underlying layers that we trained on. (e.g. weights/biases).
inf_decoder_inputs = Input(shape=(None,), name="inf_decoder_inputs")
# We'll need to force feed the two state variables into the decoder each step.
state_input_h = Input(shape=(emdedding_size,), name="state_input_h")
state_input_c = Input(shape=(emdedding_size,), name="state_input_c")
decoder_res, decoder_h, decoder_c = decoder_lstm(
    decoder_emb(inf_decoder_inputs), 
    initial_state=[state_input_h, state_input_c])
inf_decoder_out = decoder_d2(decoder_d1(decoder_res))
inf_model = Model(inputs=[inf_decoder_inputs, state_input_h, state_input_c], 
                  outputs=[inf_decoder_out, decoder_h, decoder_c])


def sentence_to_vector(sentence):
    # pre = preprocess_sentence(sentence)
    vec = np.zeros(input_len)
    sentence_list = [vec_model.wv.vocab[s].index for s in sentence.split(' ')]
    for i,w in enumerate(sentence_list):
        vec[i] = w
    return vec

# Given an input string, an encoder model (infenc_model) and a decoder model (infmodel),
# return a translated string.
def translate(input_sentence, infenc_model, infmodel, attention=False):
    sv = sentence_to_vector(input_sentence)
    # Reshape so we can use the encoder model. New shape=[samples,sequence length]
    sv = sv.reshape(1,len(sv))
    [emb_out, sh, sc] = infenc_model.predict(x=sv)
    
    i = 0
    start_vec = vec_model.wv.vocab["<start>"].index
    stop_vec = vec_model.wv.vocab["<end>"].index
    # We will continuously feed cur_vec as an input into the decoder to produce the next word,
    # which will be assigned to cur_vec. Start it with "<start>".
    cur_vec = np.zeros((1,1))
    cur_vec[0,0] = start_vec
    cur_word = "<start>"
    output_sentence = ""
    # Start doing the feeding. Terminate when the model predicts an "<end>" or we reach the end
    # of the max target language sentence length.
    while cur_word != "<end>" and i < (output_len-1):
        i += 1
        if cur_word != "<start>":
            output_sentence = output_sentence + " " + cur_word
        x_in = [cur_vec, sh, sc]
        # This will allow us to accomodate attention models, which we will talk about later.
        if attention:
            x_in += [emb_out]
        [nvec, sh, sc] = infmodel.predict(x=x_in)
        # The output of the model is a massive softmax vector with one spot for every possible word. Convert
        # it to a word ID using argmax().
        cur_vec[0,0] = np.argmax(nvec[0,0])
        cur_word = vec_model.wv.index2word[np.argmax(nvec[0,0])]
    return output_sentence

    # Let's test out the model! Feel free to modify as you see fit. Note that only words
# that we've trained the model on will be available, otherwise you'll get an error.
print(translate('וכן מתבאר מדברי הרדב"ז בתשובה ח"ז סימן ל"ב שהרמב"ם אוסר להתייחד עם אחותו', encoder_model, inf_model))
# print(translate("I am hungry", encoder_model, inf_model))
# print(translate("I know what you said.", encoder_model, inf_model))