import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Embedding, Bidirectional

class Encoder(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim,pretrained_weights, maxlen, enc_units, batch_sz):
    super(Encoder, self).__init__()
    self.batch_sz = batch_sz
    self.enc_units = enc_units
    self.embedding = Embedding(vocab_size, embedding_dim,weights=[pretrained_weights],input_length=maxlen,trainable=True)
    self.lstm = Bidirectional(LSTM(self.enc_units, return_sequences=True, return_state=True))

  def call(self, x, forward_h, forward_c, backward_h, backward_c):
    x = self.embedding(x)
    output, st_forward_h, st_forward_c, st_backward_h, st_backward_c = self.lstm(x, initial_state = [forward_h, forward_c, backward_h, backward_c])
    return output, st_forward_h, st_forward_c, st_backward_h, st_backward_c

  def initialize_hidden_state(self):
    return tf.zeros((self.batch_sz, self.enc_units)), tf.zeros((self.batch_sz, self.enc_units)), tf.zeros((self.batch_sz, self.enc_units)), tf.zeros((self.batch_sz, self.enc_units))