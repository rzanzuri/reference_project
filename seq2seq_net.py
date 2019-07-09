import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import LSTM, Dense, Embedding, Bidirectional, Concatenate
from Python_lib.Encoder import Encoder as Encoder
from Python_lib.Decoder import Decoder as Decoder
from Python_lib.BahdanauAttention import BahdanauAttention as BahdanauAttention
import Python_lib.dataSets as dataSets
import Python_lib.vectorsModel as vectorsModel

import unicodedata
import re
import numpy as np
import os
import time
import datetime
import multiprocessing

start_run = datetime.datetime.now()
print("start:", start_run)

corpus_path = "./Data/RabannyText/"
vec_model_root_path = "./VecModels/"

start_seq = '<start>'
end_seq = '<end>'
start_ref = '<start_ref>'
end_red = '<end_ref>'

iters = 6
min_count = 60
vec_size = 300 
win_size = 10
workers = multiprocessing.cpu_count() 
epochs = 1
num_examples = 1
test_size = 0.2
max_len_sent = 100
batch_size = 1


vec_model = vectorsModel.get_model_vectors(corpus_path, vec_model_root_path, win_size, iters, min_count, vec_size, workers)

sent = (start_seq + ' ') * 30
vec_model.build_vocab([sent.split()], update=True)

sent = (end_seq + ' ')* 30
vec_model.build_vocab([sent.split()], update=True)

sent = (start_ref + ' ') * 30
vec_model.build_vocab([sent.split()], update=True)

sent = (end_red + ' ') * 30
vec_model.build_vocab([sent.split()], update=True)

sentences, answers = dataSets.get_sentences_and_answers(corpus_path, max_len_sent = max_len_sent, limit= num_examples)

X_train, X_test = dataSets.get_data_set(sentences,vec_model, test_size)
Y_train, Y_test = dataSets.get_data_set(answers, vec_model, test_size)

pretrained_weights = vec_model.wv.syn0
vocab_size, emdedding_size = pretrained_weights.shape

print('Result embedding shape:', pretrained_weights.shape)
print('train_x shape:', X_train.shape)
print('train_y shape:', Y_train.shape)

steps_per_epoch = len(X_train)//batch_size
units = emdedding_size
vocab_inp_size = vocab_size
vocab_tar_size = vocab_size
path_to_file = corpus_path

def loss_function(real, pred):
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  loss_ = loss_object(real, pred)

  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask

  return tf.reduce_mean(loss_)

@tf.function
def train_step(inp, targ, forward_h, forward_c, backward_h, backward_c):
  loss = 0

  with tf.GradientTape() as tape:
    enc_output, enc_forward_h, enc_forward_c, enc_backward_h, enc_backward_c = encoder(inp, forward_h, forward_c, backward_h, backward_c)

    dec_forward_h, dec_forward_c, dec_backward_h, dec_backward_c = enc_forward_h, enc_forward_c, enc_backward_h, enc_backward_c

    dec_input = tf.expand_dims([vec_model.wv.vocab[start_seq].index] * batch_size, 1)

    # Teacher forcing - feeding the target as the next input
    for t in range(1, targ.shape[1]):
      # passing enc_output to the decoder
      predictions, dec_forward_h, dec_forward_c, dec_backward_h, dec_backward_c, _ = decoder(dec_input, dec_forward_h, dec_forward_c, dec_backward_h, dec_backward_c, enc_output)

      loss += loss_function(targ[:, t], predictions)

      # using teacher forcing
      dec_input = tf.expand_dims(targ[:, t], 1)

  batch_loss = (loss / int(targ.shape[1]))

  variables = encoder.trainable_variables + decoder.trainable_variables

  gradients = tape.gradient(loss, variables)

  optimizer.apply_gradients(zip(gradients, variables))

  return batch_loss

def evaluate(sentence):
    attention_plot = np.zeros((max_length_targ, max_length_inp))

    inputs = [vec_model.wv.vocab[i].index for i in sentence.split(' ')]
    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],
                                                           maxlen=max_length_inp,
                                                           padding='post')
    inputs = tf.convert_to_tensor(inputs)

    result = ''
    forward_h = forward_c = backward_h = backward_c = tf.zeros((1, units))

    enc_out, enc_forward_h, enc_forward_c, enc_backward_h, enc_backward_c = encoder(inputs, forward_h, forward_c, backward_h, backward_c)

    dec_forward_h, dec_forward_c, dec_backward_h, dec_backward_c = enc_forward_h, enc_forward_c, enc_backward_h, enc_backward_c


    dec_input = tf.expand_dims([vec_model.wv.vocab[start_seq].index], 0)

    for t in range(max_length_targ):
        predictions, dec_forward_h, dec_forward_c, dec_backward_h, dec_backward_c, attention_weights = decoder(dec_input,
                                                                                                               dec_forward_h, dec_forward_c, dec_backward_h, dec_backward_c,
                                                                                                               enc_out)

        # storing the attention weights to plot later on
        attention_weights = tf.reshape(attention_weights, (-1, ))
        attention_plot[t] = attention_weights.numpy()
        # print("evaluate attention_plot[",t,"]",attention_plot[t])

        predicted_id = tf.argmax(predictions[0]).numpy()

        result += vec_model.wv.index2word[predicted_id] + ' '

        if vec_model.wv.index2word[predicted_id] == end_seq:
            return result, sentence, attention_plot

        # the predicted ID is fed back into the model
        dec_input = tf.expand_dims([predicted_id], 0)

    return result, sentence, attention_plot

# function for plotting the attention weights
# function for plotting the attention weights
def plot_attention(attention, sentence, predicted_sentence):
    # print("plot_attention attention:", attention)
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(attention, cmap='viridis')

    fontdict = {'fontsize': 14}

    ax.set_xticklabels([''] + sentence, fontdict=fontdict, rotation=90)
    ax.set_yticklabels([''] + predicted_sentence, fontdict=fontdict)

    plt.show()

def translate(sentence):
    result, sentence, attention_plot = evaluate(sentence)

    print('Input: %s' % (sentence))
    print('Predicted translation: {}'.format(result))

    attention_plot = attention_plot[:len(result.split(' ')), :len(sentence.split(' '))]
    plot_attention(attention_plot, sentence.split(' '), result.split(' '))

# Calculate max_length of the target tensors
max_length_targ, max_length_inp = X_train.shape[1], Y_train.shape[1]


dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train)).shuffle(len(X_train))
dataset = dataset.batch(batch_size, drop_remainder=True)


example_input_batch, example_target_batch = next(iter(dataset))
example_input_batch.shape, example_target_batch.shape

encoder = Encoder(vocab_inp_size, emdedding_size, pretrained_weights, max_length_inp ,units, batch_size)

# sample input
forward_h, forward_c, backward_h, backward_c = encoder.initialize_hidden_state()
sample_output, forward_h, forward_c, backward_h, backward_c = encoder(example_input_batch, forward_h, forward_c, backward_h, backward_c)
print ('Encoder output shape: (batch size, sequence length, units) {}'.format(sample_output.shape))
print ('Encoder Hidden forward_h state shape: (batch size, units) {}'.format(forward_h.shape))
print ('Encoder Hidden forward_c state shape: (batch size, units) {}'.format(forward_c.shape))
print ('Encoder Hidden backward_h state shape: (batch size, units) {}'.format(backward_h.shape))
print ('Encoder Hidden backward_c state shape: (batch size, units) {}'.format(backward_c.shape))

attention_layer = BahdanauAttention(10)
attention_result, attention_weights = attention_layer(forward_h, forward_c, backward_h, backward_c, sample_output)

print("Attention result shape: (batch size, units) {}".format(attention_result.shape))
print("Attention weights shape: (batch_size, sequence_length, 1) {}".format(attention_weights.shape))

decoder = Decoder(vocab_tar_size, emdedding_size, pretrained_weights, max_length_targ ,units, batch_size)

sample_decoder_output, _, _, _, _, _ = decoder(tf.random.uniform((batch_size, 1)),
                                      forward_h, forward_c, backward_h, backward_c, sample_output)

print ('Decoder output shape: (batch_size, vocab size) {}'.format(sample_decoder_output.shape))

optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 encoder=encoder,
                                 decoder=decoder)

for epoch in range(epochs):
  start = time.time()

  enc_forward_h, enc_forward_c, enc_backward_h, enc_backward_c = encoder.initialize_hidden_state()
  total_loss = 0

  for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
    batch_loss = train_step(inp, targ, enc_forward_h, enc_forward_c, enc_backward_h, enc_backward_c)
    total_loss += batch_loss

    if batch % 100 == 0:
        print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                     batch,
                                                     batch_loss.numpy()))
  # saving (checkpoint) the model every 2 epochs
  if (epoch + 1) % 2 == 0:
    checkpoint.save(file_prefix = checkpoint_prefix)

  print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                      total_loss / steps_per_epoch))
  print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

# restoring the latest checkpoint in checkpoint_dir
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

translate(u'וכן מתבאר מדברי הרדב"ז בתשובה ח"ז סימן ל"ב שהרמב"ם אוסר להתייחד עם אחותו')