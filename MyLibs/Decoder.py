import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Embedding, Bidirectional
from MyLibs.BahdanauAttention import BahdanauAttention as BahdanauAttention

class Decoder(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, pretrained_weights, maxlen, dec_units, batch_sz):
    super(Decoder, self).__init__()
    self.batch_sz = batch_sz
    self.dec_units = dec_units
    self.embedding = Embedding(vocab_size, embedding_dim,weights=[pretrained_weights],input_length=maxlen,trainable=True)
    self.lstm = Bidirectional(LSTM(self.dec_units, return_sequences=True, return_state=True))
    self.fc = Dense(vocab_size)

    # used for attention
    self.attention = BahdanauAttention(self.dec_units)

  def call(self, x, forward_h, forward_c, backward_h, backward_c, enc_output):
    # enc_output shape == (batch_size, max_length, hidden_size)
    context_vector, attention_weights = self.attention(forward_h, forward_c, backward_h, backward_c, enc_output)

    # x shape after passing through embedding == (batch_size, 1, embedding_dim)
    x = self.embedding(x)

    # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
    x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

    # passing the concatenated vector to the GRU
    output, st_forward_h, st_forward_c, st_backward_h, st_backward_c = self.lstm(x)

    # output shape == (batch_size * 1, hidden_size)
    output = tf.reshape(output, (-1, output.shape[2]))

    # output shape == (batch_size, vocab)
    x = self.fc(output)
    return x, st_forward_h, st_forward_c, st_backward_h, st_backward_c, attention_weights
