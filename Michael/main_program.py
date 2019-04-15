import my_py_lib.dataSets as dataSets
import my_py_lib.vectorsModel as vectorsModel
import my_py_lib.NER as NER
import my_py_lib.textHandler as textHandler
from os.path import exists
import numpy as np
from nltk.tokenize import regexp_tokenize

from keras.models import Model
from keras.layers import Input, LSTM, Dense, Embedding

#model setup
vec_model_root_path = "D:\\ReferenceProject\\reference_project\\Michael\\VecModels"
curpus_path = "D:\\ReferenceProject\\reference_project\\Michael\\fra-eng-letters"

iters = 500
min_count = 1
vec_size = 300 
workers = 10 
batch_size = 64 
epochs = 500
latent_dim = 256
limit = 100000
by_char = 1
start_char = "$"
end_char = ";"
 
vec_model = vectorsModel.get_model_vectors(curpus_path ,vec_model_root_path, iters, min_count,vec_size,workers)

sentences, answers = dataSets.get_sentences_and_answers(curpus_path , limit= limit)

tag_ans = []

for ans in answers:
    tag_ans.append(start_char + ans)

data_sent = dataSets.get_data_set(sentences,vec_model,vec_size,by_char)
data_tag_ans = dataSets.get_data_set(tag_ans,vec_model,vec_size,by_char)

max_len_sent = data_sent.shape[1]
max_len_ans = data_tag_ans.shape[1]

data_ans = dataSets.get_data_set(answers,vec_model,vec_size, by_char, max_len_ans)




# Define an input sequence and process it.
encoder_inputs = Input(shape=(None,vec_size))
encoder = LSTM(latent_dim, return_state=True, return_sequences=True,)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
# We discard `encoder_outputs` and only keep the states.
encoder_states = [state_h, state_c]

# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(None, vec_size))
# We set up our decoder to return full output sequences,
# and to return internal states as well. We don't use the
# return states in the training model, but we will use them in inference.
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True )
decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                     initial_state=encoder_states)
decoder_dense = Dense(vec_size, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)



# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
# Run training
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
model.fit([data_sent, data_tag_ans], data_ans,
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

def decode_sequence_by_char(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, vec_size))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0]  = vec_model.wv[start_char]

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

        # Sample a token
        sampled_token = output_tokens[0, -1, :]
        sampled_char = vec_model.most_similar(positive=[sampled_token],topn=1)[0][0]
        decoded_sentence += sampled_char

        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_char == end_char or
           len(decoded_sentence) > max_len_ans):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, vec_size))
        target_seq[0, 0] = vec_model.wv[sampled_char]

        # Update states
        states_value = [h, c]

    return decoded_sentence

def decode_sequence(input_seq,by_char = 0):
    if by_char:
        return decode_sequence_by_char(input_seq)

    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, vec_size))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0, 5] = 1.

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

        # Sample a token
        sampled_token = output_tokens[0, -1, :]
        sampled_char = vec_model.most_similar(positive=[sampled_token],topn=1)[0][0]
        if(sampled_char != end_char):
            decoded_sentence += sampled_char + " "
        else:
            decoded_sentence += sampled_char

        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_char == end_char or
           len(regexp_tokenize(decoded_sentence, pattern=r'\w+|\$[\d\.]+|\S+')) > max_len_ans):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, vec_size))
        target_seq[0, 0] = sampled_token

        # Update states
        states_value = [h, c]

    return decoded_sentence

for seq_index in range(20):
    # Take one sequence (part of the training set)
    # for trying out decoding.
    input_seq = data_sent[seq_index: seq_index + 1]
    decoded_sentence = decode_sequence(input_seq, by_char)
    print('---------------------------------------------')
    print('Input sentence:', sentences[seq_index])
    print('Decoded sentence:', decoded_sentence)