import tensorflow as tf
from tensorflow.python.keras.layers.embeddings import Embedding

# GRU layer
class GRU(tf.keras.layers.Layer):
  '''
    Arguments:
      units (int): hidden dimension 
      inp_shape (int): Embedding dimension 
    Output:
      h_t (Tensor): 
        Current hidden state
        shape=(None, units) 
  '''

  def __init__(self, units, inp_shape):
    super(GRU, self).__init__()
    self.units = units
    self.inp_shape = inp_shape
    self.W = self.add_weight("W", shape=(3, self.units, self.inp_shape))
    self.U = self.add_weight("U", shape=(3, self.inp_shape, self.units))
    self.b = self.add_weight("b", shape=(self.units, 1), initializer="zeros", trainable=True)
    
  def call(self, pre_h, x):

    # Update gate: Decide how much the unit updates its activation, or content
    z_t = tf.nn.sigmoid(
        tf.matmul(x, tf.transpose(self.W[0])) + tf.matmul(pre_h, tf.transpose(self.U[0])) + self.b[0])

    # Reset gate: Forget the previously state
    r_t = tf.nn.sigmoid(
        tf.matmul(x, tf.transpose(self.W[1])) + tf.matmul(pre_h, tf.transpose(self.U[1])) + self.b[1])

    # Current memory content
    h_proposal = tf.nn.tanh(
        tf.matmul(x, tf.transpose(self.W[2])) + tf.matmul(tf.multiply(r_t, pre_h), tf.transpose(self.U[2])) + self.b[2])

    # Current hidden state
    h_t = tf.multiply((1 - z_t), pre_h) + tf.multiply(z_t, h_proposal)
    
    return h_t

# Define GRU model
class GRU_RNN(tf.keras.Model):
  '''
    Arguments
      units (int):
      embedding_size (int):
      vocab_size (int):
      input_length (int):
      num_class (int):
    Output
  '''
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
        tf.keras.layers.Dense(32, input_shape=(units,), activation="elu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(num_class, activation='softmax')
    ])

  def call(self, sentence):
    '''
      sentence
    '''
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
