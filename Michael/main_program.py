import my_py_lib.dataSets as dataSets
import my_py_lib.vectorsModel as vectorsModel
import my_py_lib.NER as NER
import my_py_lib.textHandler as textHandler
from os.path import exists
import numpy as np
from nltk.tokenize import regexp_tokenize

from keras.models import Model
from keras.layers import Input, LSTM, Dense, Embedding
def gen_fr_ans(sen):
    return "\t" + sen
#vecmodel_setup
vec_model_root_path = "D:\\FinalProject\\VecModels\\"
    
#vec_model_name = "Wiki_en"
# vec_model_name = "fra-eng"
vec_model_name = "fra.txt"

model_vec_text_path = "D:\\FinalProject\\" + vec_model_name + ".vecmodel"
iters = 10
min_count = 1
vec_size = 300 
workers = 4 
batch_size = 64 
epochs = 10
latent_dim = 256

#text_file_setup
text_root_path = "D:\\FinalProject\\fra-eng\\"
text_file_name = "eng.fra.txt"
 
vec_model_eng = vectorsModel.get_model_vectors(vec_model_root_path + "eng.txt.vecmodel",vec_model_root_path, iters, min_count,vec_size,workers)
vec_model_fr = vectorsModel.get_model_vectors(vec_model_root_path + "fr.txt.vecmodel",vec_model_root_path, iters, min_count,vec_size,workers)

sentences, answers = dataSets.get_sentences_and_answers(text_root_path + text_file_name + ".ans", get_answer_func= gen_fr_ans, limit= 50)

data_sent = dataSets.get_data_set(sentences,vec_model_eng,vec_size)
data_ans = dataSets.get_data_set(answers,vec_model_fr,vec_size)

# Define an input sequence and process it.
encoder_inputs = Input(shape=(None, vec_size))
encoder = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
# We discard `encoder_outputs` and only keep the states.
encoder_states = [state_h, state_c]

# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(None, vec_size))
# We set up our decoder to return full output sequences,
# and to return internal states as well. We don't use the
# return states in the training model, but we will use them in inference.
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                     initial_state=encoder_states)
decoder_dense = Dense(vec_size, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

# Run training
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
model.fit([data_sent, data_ans], data_ans,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.2)
# Save model
model.save('s2s.h5')
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
    target_seq = np.zeros((1, 1, vec_size))
    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

        # Sample a token
        sampled_token = output_tokens[0, -1, :]
        sampled_word = vec_model_fr.most_similar(positive=[sampled_token],topn=1)[0][0]
        decoded_sentence += sampled_word

        # Exit condition: either hit max length
        # or find stop character.
        if (len(regexp_tokenize(decoded_sentence, pattern=r'\w+|\$[\d\.]+|\S+')) >= data_ans.shape[1]):
            stop_condition = True
     
        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, vec_size))
        target_seq[0, 0] = vec_model_fr.wv[sampled_word]

        # Update states
        states_value = [h, c]

    return decoded_sentence


for seq_index in range(20):
    # Take one sequence (part of the training set)
    # for trying out decoding.
    input_seq = data_sent[seq_index: seq_index + 1]
    print('-')
    print('Input sentence:', sentences[seq_index])
    decoded_sentence = decode_sequence(input_seq)
    print('Decoded sentence:', decoded_sentence)