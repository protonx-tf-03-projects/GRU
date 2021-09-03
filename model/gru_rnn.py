import tensorflow as tf
from tensorflow.python.keras.layers.embeddings import Embedding

# GRU layer
class GRU(tf.keras.layers.Layer):
  '''
    Arguments:
    units: 
    input_shape
  '''

  def __init__(self, units, _input_shape):
    super(GRU, self).__init__()
    self.units = units
    self._input_shape = _input_shape
    self.W = self.add_weight("W", shape=(3, self.units, self._input_shape))
    self.U = self.add_weight("U", shape=(3, self.units, self.units))

  def call(self, pre_h, x):

    # Update gate: Decide how much the unit updates its activation, or content
    z_t = tf.nn.sigmoid(
        tf.matmul(x, tf.transpose(self.W[0])) + tf.matmul(pre_h, tf.transpose(self.U[0])))

    # Reset gate: Forget the previously state
    r_t = tf.nn.sigmoid(
        tf.matmul(x, tf.transpose(self.W[1])) + tf.matmul(pre_h, tf.transpose(self.U[1])))

    # Current memory content
    h_proposal = tf.nn.tanh(
        tf.matmul(x, tf.transpose(self.W[2])) + tf.matmul(tf.multiply(r_t, pre_h), tf.transpose(self.U[2])))

    # Current hidden state
    h_t = tf.multiply((1 - z_t), pre_h) + tf.multiply(z_t, h_proposal)

    return h_t

# Define GRU model
class GRU_RNN(tf.keras.Model):
  def __init__(self, units, embedding_size, vocab_size, input_length, num_class):
    super(GRU_RNN, self).__init__()
    self.input_length = input_length
    self.units = units
    self.num_class = num_class

    # Embedding
    self.embedding = tf.keras.layers.Embedding(
      vocab_size,
      embedding_size,
      input_length=input_length
    )

    # Using gru cell
    self.model = GRU(units, embedding_size)

    # Pass each hidden state through Rnn basic
    self.classification_layer = tf.keras.models.Sequential([
        tf.keras.layers.Dense(32, input_shape=(units,), activation="swish"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(num_class, activation='softmax')
    ])

  def call(self, sentence):

    batch_size = tf.shape(sentence)[0]

    # Initial hidden_state
    pre_h = tf.zeros([batch_size, self.units])

    # embedded_sentence: (batch_size, input_length, embedding_size)
    embedded_sentence = self.embedding(sentence)
    
    for i in range(self.input_length):
      word = embedded_sentence[:, i, :]
      pre_h = self.model(pre_h, word)
    
    h = pre_h

    # Predition by lastest hidden_state
    output = self.classification_layer(h)
    print("===output_layer===", output)

    return output
