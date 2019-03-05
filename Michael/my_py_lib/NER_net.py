from __future__ import print_function

from keras.models import Model
from keras.layers import Input, LSTM, Dense
import numpy as np
from os.path import exists
import gensim
from gensim.models import word2vec
from nltk.tokenize import regexp_tokenize, wordpunct_tokenize, blankline_tokenize


batch_size = 64  # Batch size for training.
epochs = 5  # Number of epochs to train for.
latent_dim = 256  # Latent dimensionality of the encoding space.
num_samples = 100000  # Number of samples to train on.
# Path to the data txt file on disk.
data_path = 'fra-eng/fra.txt'

# Vectorize the data.
input_texts = []
target_texts = []
input_words = set()
target_words = set()
with open(data_path, 'r', encoding='utf-8') as f:
    lines = f.read().split('\n')
for line in lines[: min(num_samples, len(lines) - 1)]:
    input_text, target_text = line.split('\t')
    # We use "tab" as the "start sequence" character
    # for the targets, and "\n" as "end sequence" character.
    target_text = target_text
    input_texts.append(input_text)
    target_texts.append(target_text)

    # for word in regexp_tokenize(input_text, pattern=r'\w+|\$[\d\.]+|\S+'):
    #     if word not in input_words:
    #         input_words.add(word)        

    # for word in regexp_tokenize(target_text, pattern=r'\w+|\$[\d\.]+|\S+'):
    #     if word not in target_words:
    #         target_words.add(word)    

input_words = sorted(list(input_words))
target_words = sorted(list(target_words))
num_encoder_tokens = len(input_words)
num_decoder_tokens = len(target_words)

max_encoder_seq_length = 0
max_decoder_seq_length = 0

for txt in input_texts:
    txt_len = len(regexp_tokenize(txt, pattern=r'\w+|\$[\d\.]+|\S+'))
    # if(len(txt) > 1 and (txt[len(txt) - 1] == "." or txt[len(txt) - 1] == "," or txt[len(txt) - 1] == "!" or txt[len(txt) - 1] == "?")):
    #     txt_len = txt_len + 1
    max_encoder_seq_length = max(max_encoder_seq_length,txt_len)

for txt in target_texts:
    txt_len = len(regexp_tokenize(txt, pattern=r'\w+|\$[\d\.]+|\S+'))
    # if(len(txt) > 1 and (txt[len(txt) - 1] == "." or txt[len(txt) - 1] == "," or txt[len(txt) - 1] == "!" or txt[len(txt) - 1] == "?")):
    #     txt_len = txt_len + 1
    max_decoder_seq_length = max(max_decoder_seq_length,txt_len)

num_encoder_tokens = 300
num_decoder_tokens = 300
print('Number of samples:', len(input_texts))
print('Number of unique input tokens:', num_encoder_tokens)
print('Number of unique output tokens:', num_decoder_tokens)
print('Max sequence length for inputs:', max_encoder_seq_length)
print('Max sequence length for outputs:', max_decoder_seq_length)
#vecmodel_setup
vec_model_root_path = "D:\\FinalProject\\VecModels\\fra.txt.vecmodel"
    
model_vec = gensim.models.Word2Vec.load(vec_model_root_path)
# input_token_index = dict(
#     [(word, i) for i, word in enumerate(input_words)])
# target_token_index = dict(
#     [(word, i) for i, word in enumerate(target_words)])

encoder_input_data = np.zeros(
    (len(input_texts), max_encoder_seq_length, num_encoder_tokens),
    dtype='float32')
decoder_input_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
    dtype='float32')
decoder_target_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
    dtype='float32')

for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):    
    for t, char in enumerate(regexp_tokenize(input_text, pattern=r'\w+|\$[\d\.]+|\S+')):
        if char not in model_vec.wv.vocab:
            encoder_input_data[i, t] = np.zeros(300)
        else:
            encoder_input_data[i, t] = model_vec.wv[char]
    for t, char in enumerate(regexp_tokenize(target_text, pattern=r'\w+|\$[\d\.]+|\S+')):
        # decoder_target_data is ahead of decoder_input_data by one timestep
        if char not in model_vec.wv.vocab:
            decoder_input_data[i, t] = np.zeros(300)
            decoder_target_data[i, t] = np.zeros(300)
        else :       
            decoder_input_data[i, t] = model_vec.wv[char]
            decoder_target_data[i, t] = model_vec.wv[char]           

# Define an input sequence and process it.
encoder_inputs = Input(shape=(None, num_encoder_tokens))
encoder = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
# We discard `encoder_outputs` and only keep the states.
encoder_states = [state_h, state_c]

# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(None, num_decoder_tokens))
# We set up our decoder to return full output sequences,
# and to return internal states as well. We don't use the
# return states in the training model, but we will use them in inference.
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                     initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

# Run training
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.2)
# Save model
model.save('s2s.h5')

# Next: inference mode (sampling).
# Here's the drill:
# 1) encode input and retrieve initial decoder state
# 2) run one step of decoder with this initial state
# and a "start of sequence" token as target.
# Output will be the next target token
# 3) Repeat with the current target token and current states

# Define sampling models
encoder_model = Model(encoder_inputs, encoder_states)

decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs, state_h, state_c = decoder_lstm(
    decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)


def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0] = model

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char

        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_char == '\n' or
           len(decoded_sentence) > max_decoder_seq_length):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.

        # Update states
        states_value = [h, c]

    return decoded_sentence


for seq_index in range(100):
    # Take one sequence (part of the training set)
    # for trying out decoding.
    input_seq = encoder_input_data[seq_index: seq_index + 1]
    decoded_sentence = decode_sequence(input_seq)
    print('-')
    print('Input sentence:', input_texts[seq_index])
    print('Decoded sentence:', decoded_sentence)