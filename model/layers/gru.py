import tensorflow as tf

class GRU(tf.keras.layers.Layer):
  '''
    Arguments:
  '''
  def __init__(self, units, inp_shape):
    super(GRU, self).__init__()
    self.units = units
    self.inp_shape = inp_shape
    self.W = self.add_weight("W", shape=(3, self.units, self.inp_shape))
    self.U = self.add_weight("U", shape=(3, self.units, self.units))
    # self.b = self.add_weight("b", shape=(3, self.units, self.units))

  def call(self, pre_layer, x):
    pre_h, pre_c = tf.unstack(pre_layer)

    # TODO: Update gate: Decide how much the unit updates its activation, or content
    z_t = tf.nn.sigmoid(
        tf.matmul(x, tf.transpose(self.W[0])) + tf.matmul(pre_h, tf.transpose(self.U[0])))

    # TODO: Reset gate: Forget the previously state
    r_t = tf.nn.sigmoid(
        tf.matmul(x, tf.transpose(self.W[1])) + tf.matmul(pre_h, tf.transpose(self.U[1])))

    # TODO: Current memory content
    h_proposal = tf.nn.tanh(
        tf.matmul(x, tf.transpose(self.W[2])) + tf.matmul(tf.multiply(r_t, pre_h), tf.transpose(self.U[2])))

    # TODO: Output gate (at time t): Linear interpolation between the previous and candidate activation
    h_t = tf.multiply((1 - z_t), pre_h) + tf.multiply(z_t, h_proposal)

    return h_t


