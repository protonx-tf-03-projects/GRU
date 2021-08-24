import tensorflow as tf

class Tanh(tf.keras.layers.Layer):
    """
        Using traditional RNN but the bounded function is a tanh function
    """
    def __init__(self, units, inp_shape):
        super(Tanh,self).__init__()
        self.units = units
        self.inp_shape = inp_shape
        self.W = self.add_weight("W", shape=(1, self.units, self.inp_shape))
        self.U = self.add_weight("U", shape=(1, self.units, self.units))

    def call(self, pre_layer, x):
        # pre_h, pre_c = tf.unstack(pre_layer)
        h = tf.nn.tanh(tf.matmul(x, tf.transpose(self.W[0])) + tf.matmul(pre_layer, tf.transpose(self.U[0])))
        return h

