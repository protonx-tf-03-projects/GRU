import tensorflow as tf
import pandas as np
from layers.gru import GRU

# TODO: Define call accuracy
def cal_acc(real, pred):
  pass

# TODO: GRU RNN model
class GRU_RNN(tf.keras.Model):
  def __init__(self, units, embedding_size, vocab_size, input_length):
    super(GRU_RNN, self).__init__()
    self.input_length = input_length
    self.units = units

    self.embedding = tf.keras.layers.Embedding(
      vocab_size,
      embedding_size,
      input_length=input_length
    )

    self.gru = GRU(units, embedding_size)

    self.classfication_model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(64, input_shape=(units,)),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

  def call(self, sentence):

    # TODO: Init parameters

    # TODO
    return None


  def initialize_parameters(n, d_1, d_2, num_classes):
    np.random.seed(42)
    W1 = np.random.randn(d_1, n) * 0.01
    U1 = np.random.randn(d_1, n) * 0.01
    b1 = np.zeros((d_1, 1))
    W2 = np.random.randn(d_2, d_1) * 0.01
    U2 = np.random.randn(d_2, d_1) * 0.01
    b2 = np.zeros((d_2, 1))
    W3 = np.random.randn(num_classes, d_2) * 0.01
    U3 = np.random.randn(num_classes, d_2) * 0.01
    b3 = np.zeros((num_classes, 1))

    parameters = {"W1": W1,
                  "U1": U1,
                  "b1": b1,
                  "W2": W2,
                  "U2": U2,
                  "b2": b2,
                  "W3": W3,
                  "U3": U3,
                  "b3": b3
                  }

    for param in parameters:
      param.attach_grad()

    return parameters
      

   
