from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding, Flatten, TimeDistributed, Dropout, LSTMCell, RNN
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.utils import tf_utils
from tensorflow.keras import backend as K
import tensorflow as tf

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

class LanguageIndex():
    def __init__(self, lang):
        self.lang = lang
        self.word2idx = {}
        self.idx2word = {}
        self.vocab = set()
        self.create_index()
    def create_index(self):
        for phrase in self.lang:
            self.vocab.update(phrase.split(' '))
        self.vocab = sorted(self.vocab)
        self.word2idx["<pad>"] = 0
        self.idx2word[0] = "<pad>"
        for i,word in enumerate(self.vocab):
            self.word2idx[word] = i + 1
            self.idx2word[i+1] = word

# # Download the file
# path_to_zip = tf.keras.utils.get_file(
#     'spa-eng.zip', origin='http://download.tensorflow.org/data/spa-eng.zip', 
#     extract=True)
path_to_file = "./Data/spa-eng/spa.txt"

def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')
def preprocess_sentence(w):
    w = unicode_to_ascii(w.lower().strip())
    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)
    w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)
    w = w.rstrip().strip()
    w = "<start> " + w + " <end>"
    return w

def max_length(t):
    return max(len(i) for i in t)

def create_dataset(path, num_examples):
    lines = open(path, encoding="UTF-8").read().strip().split("\n")
    word_pairs = [[preprocess_sentence(w) for w in l.split("\t")] for l in lines[:num_examples]]
    return word_pairs

def load_dataset(path, num_examples):
    pairs = create_dataset(path, num_examples)
    out_lang = LanguageIndex(sp for en, sp in pairs)
    in_lang = LanguageIndex(en for en, sp in pairs)
    input_data = [[in_lang.word2idx[s] for s in en.split(' ')] for en, sp in pairs]
    output_data = [[out_lang.word2idx[s] for s in sp.split(' ')] for en, sp in pairs]

    max_length_in, max_length_out = max_length(input_data), max_length(output_data)
    input_data = tf.keras.preprocessing.sequence.pad_sequences(input_data, maxlen=max_length_in, padding="post")
    output_data = tf.keras.preprocessing.sequence.pad_sequences(output_data, maxlen=max_length_out, padding="post")
    return input_data, output_data, in_lang, out_lang, max_length_in, max_length_out

    #num_examples = 118000 # Full example set.
num_examples = 30000 # Partial set for faster training
input_data, teacher_data, input_lang, target_lang, len_input, len_target = load_dataset(path_to_file, num_examples)


target_data = [[teacher_data[n][i+1] for i in range(len(teacher_data[n])-1)] for n in range(len(teacher_data))]
target_data = tf.keras.preprocessing.sequence.pad_sequences(target_data, maxlen=len_target, padding="post")
print(target_data.shape)
target_data = target_data.reshape((target_data.shape[0], target_data.shape[1], 1))

# Shuffle all of the data in unison. This training set has the longest (e.g. most complicated) data at the end,
# so a simple Keras validation split will be problematic if not shuffled.
p = np.random.permutation(len(input_data))
input_data = input_data[p]
teacher_data = teacher_data[p]
target_data = target_data[p]

BUFFER_SIZE = len(input_data)
BATCH_SIZE = 64
embedding_dim = 256
units = 1024
vocab_in_size = len(input_lang.word2idx)
vocab_out_size = len(target_lang.word2idx)

################################ INPUTS ###########################
#len_input - ????? ?? ????? ??? ???? ?????? ???? ?????? ????.
#vocab_in_size -  ????? ?? ??????, ?"? ???? ?????? ?????? ??? ?????? ?? ????.
#embedding_dim - ?? ????
#units - ?? ????
#vocab_out_size - ???? ?????? ?????? ??? ?????? ?? ????, ?????? ?? ???? ???.
#input_data - 
#teacher_data - 
#target_data - 
#BATCH_SIZE - ?? ????.
###################################################################

# Create the Encoder layers first.
encoder_inputs = Input(shape=(len_input,))
encoder_emb = Embedding(input_dim=vocab_in_size, output_dim=embedding_dim)
encoder_lstm = LSTM(units=units, return_sequences=True, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(encoder_emb(encoder_inputs))
encoder_states = [state_h, state_c]

# Now create the Decoder layers.
decoder_inputs = Input(shape=(None,))
decoder_emb = Embedding(input_dim=vocab_out_size, output_dim=embedding_dim)
decoder_lstm = LSTM(units=units, return_sequences=True, return_state=True)
decoder_lstm_out, _, _ = decoder_lstm(decoder_emb(decoder_inputs), initial_state=encoder_states)
# Two dense layers added to this model to improve inference capabilities.
decoder_d1 = Dense(units, activation="relu")
decoder_d2 = Dense(vocab_out_size, activation="softmax")
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
epochs = 10
history = model.fit([input_data, teacher_data], target_data,
                 batch_size=BATCH_SIZE,
                 epochs=epochs,
                 validation_split=0.2)

model.save("./new_net.md")
# Plot the results of the training.
plt.plot(history.history['sparse_categorical_accuracy'], label="Training loss")
plt.plot(history.history['val_sparse_categorical_accuracy'], label="Validation loss")
plt.show()

# Create the encoder model from the tensors we previously declared.
encoder_model = Model(encoder_inputs, [encoder_outputs, state_h, state_c])

# Generate a new set of tensors for our new inference decoder. Note that we are using new tensors, 
# this does not preclude using the same underlying layers that we trained on. (e.g. weights/biases).
inf_decoder_inputs = Input(shape=(None,), name="inf_decoder_inputs")
# We'll need to force feed the two state variables into the decoder each step.
state_input_h = Input(shape=(units,), name="state_input_h")
state_input_c = Input(shape=(units,), name="state_input_c")
decoder_res, decoder_h, decoder_c = decoder_lstm(
    decoder_emb(inf_decoder_inputs), 
    initial_state=[state_input_h, state_input_c])
inf_decoder_out = decoder_d2(decoder_d1(decoder_res))
inf_model = Model(inputs=[inf_decoder_inputs, state_input_h, state_input_c], 
                  outputs=[inf_decoder_out, decoder_h, decoder_c])

                  # Converts the given sentence (just a string) into a vector of word IDs
# using the language specified. This can be used for either the input (English)
# or target (Spanish) languages.
# Output is 1-D: [timesteps/words]
def sentence_to_vector(sentence, lang):
    pre = preprocess_sentence(sentence)
    vec = np.zeros(len_input)
    sentence_list = [lang.word2idx[s] for s in pre.split(' ')]
    for i,w in enumerate(sentence_list):
        vec[i] = w
    return vec

# Given an input string, an encoder model (infenc_model) and a decoder model (infmodel),
# return a translated string.
def translate(input_sentence, infenc_model, infmodel, attention=False):
    sv = sentence_to_vector(input_sentence, input_lang)
    # Reshape so we can use the encoder model. New shape=[samples,sequence length]
    sv = sv.reshape(1,len(sv))
    [emb_out, sh, sc] = infenc_model.predict(x=sv)
    
    i = 0
    start_vec = target_lang.word2idx["<start>"]
    stop_vec = target_lang.word2idx["<end>"]
    # We will continuously feed cur_vec as an input into the decoder to produce the next word,
    # which will be assigned to cur_vec. Start it with "<start>".
    cur_vec = np.zeros((1,1))
    cur_vec[0,0] = start_vec
    cur_word = "<start>"
    output_sentence = ""
    # Start doing the feeding. Terminate when the model predicts an "<end>" or we reach the end
    # of the max target language sentence length.
    while cur_word != "<end>" and i < (len_target-1):
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
        cur_word = target_lang.idx2word[np.argmax(nvec[0,0])]
    return output_sentence

    # Let's test out the model! Feel free to modify as you see fit. Note that only words
# that we've trained the model on will be available, otherwise you'll get an error.
print(translate("I love you", encoder_model, inf_model))
print(translate("I am hungry", encoder_model, inf_model))
print(translate("I know what you said.", encoder_model, inf_model))

# RNN "Cell" classes in Keras perform the actual data transformations at each timestep. Therefore, in order
# to add attention to LSTM, we need to make a custom subclass of LSTMCell.
class AttentionLSTMCell(LSTMCell):
    def __init__(self, **kwargs):
        self.attentionMode = False
        super(AttentionLSTMCell, self).__init__(**kwargs)
    
    # Build is called to initialize the variables that our cell will use. We will let other Keras
    # classes (e.g. "Dense") actually initialize these variables.
    @tf_utils.shape_type_conversion
    def build(self, input_shape):        
        # Converts the input sequence into a sequence which can be matched up to the internal
        # hidden state.
        self.dense_constant = TimeDistributed(Dense(self.units, name="AttLstmInternal_DenseConstant"))
        
        # Transforms the internal hidden state into something that can be used by the attention
        # mechanism.
        self.dense_state = Dense(self.units, name="AttLstmInternal_DenseState")
        
        # Transforms the combined hidden state and converted input sequence into a vector of
        # probabilities for attention.
        self.dense_transform = Dense(1, name="AttLstmInternal_DenseTransform")
        
        # We will augment the input into LSTMCell by concatenating the context vector. Modify
        # input_shape to reflect this.
        batch, input_dim = input_shape[0]
        batch, timesteps, context_size = input_shape[-1]
        lstm_input = (batch, input_dim + context_size)
        
        # The LSTMCell superclass expects no constant input, so strip that out.
        return super(AttentionLSTMCell, self).build(lstm_input)
    
    # This must be called before call(). The "input sequence" is the output from the 
    # encoder. This function will do some pre-processing on that sequence which will
    # then be used in subsequent calls.
    def setInputSequence(self, input_seq):
        self.input_seq = input_seq
        self.input_seq_shaped = self.dense_constant(input_seq)
        self.timesteps = tf.shape(self.input_seq)[-2]
    
    # This is a utility method to adjust the output of this cell. When attention mode is
    # turned on, the cell outputs attention probability vectors across the input sequence.
    def setAttentionMode(self, mode_on=False):
        self.attentionMode = mode_on
    
    # This method sets up the computational graph for the cell. It implements the actual logic
    # that the model follows.
    def call(self, inputs, states, constants):
        # Separate the state list into the two discrete state vectors.
        # ytm is the "memory state", stm is the "carry state".
        ytm, stm = states
        # We will use the "carry state" to guide the attention mechanism. Repeat it across all
        # input timesteps to perform some calculations on it.
        stm_repeated = K.repeat(self.dense_state(stm), self.timesteps)
        # Now apply our "dense_transform" operation on the sum of our transformed "carry state" 
        # and all encoder states. This will squash the resultant sum down to a vector of size
        # [batch,timesteps,1]
        # Note: Most sources I encounter use tanh for the activation here. I have found with this dataset
        # and this model, relu seems to perform better. It makes the attention mechanism far more crisp
        # and produces better translation performance, especially with respect to proper sentence termination.
        combined_stm_input = self.dense_transform(
            keras.activations.relu(stm_repeated + self.input_seq_shaped))
        # Performing a softmax generates a log probability for each encoder output to receive attention.
        score_vector = keras.activations.softmax(combined_stm_input, 1)
        # In this implementation, we grant "partial attention" to each encoder output based on 
        # it's log probability accumulated above. Other options would be to only give attention
        # to the highest probability encoder output or some similar set.
        context_vector = K.sum(score_vector * self.input_seq, 1)
        
        # Finally, mutate the input vector. It will now contain the traditional inputs (like the seq2seq
        # we trained above) in addition to the attention context vector we calculated earlier in this method.
        inputs = K.concatenate([inputs, context_vector])
        
        # Call into the super-class to invoke the LSTM math.
        res = super(AttentionLSTMCell, self).call(inputs=inputs, states=states)
        
        # This if statement switches the return value of this method if "attentionMode" is turned on.
        if(self.attentionMode):
            return (K.reshape(score_vector, (-1, self.timesteps)), res[1])
        else:
            return res

# Custom implementation of the Keras LSTM that adds an attention mechanism.
# This is implemented by taking an additional input (using the "constants" of the
# RNN class) into the LSTM: The encoder output vectors across the entire input sequence.
class LSTMWithAttention(RNN):
    def __init__(self, units, **kwargs):
        cell = AttentionLSTMCell(units=units)
        self.units = units
        super(LSTMWithAttention, self).__init__(cell, **kwargs)
        
    @tf_utils.shape_type_conversion
    def build(self, input_shape):
        self.input_dim = input_shape[0][-1]
        self.timesteps = input_shape[0][-2]
        return super(LSTMWithAttention, self).build(input_shape) 
    
    # This call is invoked with the entire time sequence. The RNN sub-class is responsible
    # for breaking this up into calls into the cell for each step.
    # The "constants" variable is the key to our implementation. It was specifically added
    # to Keras to accomodate the "attention" mechanism we are implementing.
    def call(self, x, constants, **kwargs):
        if isinstance(x, list):
            self.x_initial = x[0]
        else:
            self.x_initial = x
        
        # The only difference in the LSTM computational graph really comes from the custom
        # LSTM Cell that we utilize.
        self.cell._dropout_mask = None
        self.cell._recurrent_dropout_mask = None
        self.cell.setInputSequence(constants[0])
        return super(LSTMWithAttention, self).call(inputs=x, constants=constants, **kwargs)

# Below is test code to validate that this LSTM class and the associated cell create a
# valid computational graph.
test = LSTMWithAttention(units=units, return_sequences=True, return_state=True)
test.cell.setAttentionMode(True)
attenc_inputs2 = Input(shape=(len_input,))
attenc_emb2 = Embedding(input_dim=vocab_in_size, output_dim=embedding_dim)
test(inputs=attenc_emb2(attenc_inputs2), constants=attenc_emb2(attenc_inputs2), initial_state=None)

# Re-create an entirely new model and set of layers for the attention model

# Encoder Layers
attenc_inputs = Input(shape=(len_input,), name="attenc_inputs")
attenc_emb = Embedding(input_dim=vocab_in_size, output_dim=embedding_dim)
attenc_lstm = LSTM(units=units, return_sequences=True, return_state=True)
attenc_outputs, attstate_h, attstate_c = attenc_lstm(attenc_emb(attenc_inputs))
attenc_states = [attstate_h, attstate_c]

attdec_inputs = Input(shape=(None,))
attdec_emb = Embedding(input_dim=vocab_out_size, output_dim=embedding_dim)
attdec_lstm = LSTMWithAttention(units=units, return_sequences=True, return_state=True)
# Note that the only real difference here is that we are feeding attenc_outputs to the decoder now.
# Nice and clean!
attdec_lstm_out, _, _ = attdec_lstm(inputs=attdec_emb(attdec_inputs), 
                                    constants=attenc_outputs, 
                                    initial_state=attenc_states)
attdec_d1 = Dense(units, activation="relu")
attdec_d2 = Dense(vocab_out_size, activation="softmax")
attdec_out = attdec_d2(Dropout(rate=.4)(attdec_d1(Dropout(rate=.4)(attdec_lstm_out))))

attmodel = Model([attenc_inputs, attdec_inputs], attdec_out)
attmodel.compile(optimizer=tf.train.AdamOptimizer(), loss="sparse_categorical_crossentropy", metrics=['sparse_categorical_accuracy'])

epochs = 20
atthist = attmodel.fit([input_data, teacher_data], target_data,
                 batch_size=BATCH_SIZE,
                 epochs=epochs,
                 validation_split=0.2)

# Plot the results of the training.
plt.plot(atthist.history['sparse_categorical_accuracy'], label="Training loss")
plt.plot(atthist.history['val_sparse_categorical_accuracy'], label="Validation loss")
plt.show()

def createAttentionInference(attention_mode=False):
    # Create an inference model using the layers already trained above.
    attencoder_model = Model(attenc_inputs, [attenc_outputs, attstate_h, attstate_c])
    state_input_h = Input(shape=(units,), name="state_input_h")
    state_input_c = Input(shape=(units,), name="state_input_c")
    attenc_seq_out = Input(shape=attenc_outputs.get_shape()[1:], name="attenc_seq_out")
    inf_attdec_inputs = Input(shape=(None,), name="inf_attdec_inputs")
    attdec_lstm.cell.setAttentionMode(attention_mode)
    attdec_res, attdec_h, attdec_c = attdec_lstm(attdec_emb(inf_attdec_inputs), 
                                                 initial_state=[state_input_h, state_input_c], 
                                                 constants=attenc_seq_out)
    attinf_model = None
    if not attention_mode:
        inf_attdec_out = attdec_d2(attdec_d1(attdec_res))
        attinf_model = Model(inputs=[inf_attdec_inputs, state_input_h, state_input_c, attenc_seq_out], 
                             outputs=[inf_attdec_out, attdec_h, attdec_c])
    else:
        attinf_model = Model(inputs=[inf_attdec_inputs, state_input_h, state_input_c, attenc_seq_out], 
                             outputs=[attdec_res, attdec_h, attdec_c])
    return attencoder_model, attinf_model

attencoder_model, attinf_model = createAttentionInference()
print(translate("I love you", attencoder_model, attinf_model, True))
print(translate("I am hungry", attencoder_model, attinf_model, True))
print(translate("What is your name.", attencoder_model, attinf_model, True))

def investigate_attention(input_sentence, output_sentence, infenc_model, infmodel):
    sv = sentence_to_vector(input_sentence, input_lang)
    # Shape=samples,sequence length
    sv = sv.reshape(1,len(sv))
    [emb_out, sh, sc] = infenc_model.predict(x=sv)
    
    outvec = sentence_to_vector(output_sentence, target_lang)
    i = 0
    cur_vec = np.zeros((1,1))
    cur_vec[0,0] = outvec[0]
    cur_word = "<start>"
    output_attention = []
    while i < (len(outvec)-1):
        i += 1
        x_in = [cur_vec, sh, sc, emb_out]
        [nvec, sh, sc] = infmodel.predict(x=x_in)
        output_attention += [nvec]
        cur_vec[0,0] = outvec[i]
    return output_attention

def plotAttention(attMatrix):
    attMatrix = np.asarray(attMatrix)
    attMatrix = np.reshape(attMatrix, (attMatrix.shape[0], attMatrix.shape[-1]))
    #print(attMatrix)
    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(attMatrix, aspect="auto")

    plt.show()

attencoder_model, attinf_model = createAttentionInference(True)
#print(investigate_attention("I love me", attencoder_model, attinf_model, True))
#print(investigate_attention("I am hungry", attencoder_model, attinf_model, True))
plotAttention(investigate_attention("You can use a dictionary for this exam.", "Para este examen podéis usar un diccionario.", attencoder_model, attinf_model))