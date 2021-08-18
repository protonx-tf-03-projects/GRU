import tensorflow as tf

class Scratch_LSTM(tf.keras.layers.Layer):
    """
        Scratch LSTM with the equations by modifying  the original LSTM tensorflow model
    """
    def __init__(self, units, inp_shape):
        super(Scratch_LSTM, self).__init__()
        self.units = units
        self.inp_shape = inp_shape
        self.W = self.add_weight("W", shape=(4, self.units, self.inp_shape))
        self.U = self.add_weight("U", shape=(4, self.units, self.units))

    def call(self, pre_layer, x):
        pre_h, pre_c = tf.unstack(pre_layer)

        # Control the input values :  Input Gate:
        i_t = tf.nn.sigmoid(tf.matmul(x, tf.transpose(self.W[0])) + tf.matmul(pre_h, tf.transpose(self.U[0])))

        # Control the numbers of data need to keep: Forget Gate
        f_t = tf.nn.sigmoid(tf.matmul(x, tf.transpose(self.W[1])) + tf.matmul(pre_h, tf.transpose(self.U[1])))

        # Control the numbers of data in output: Output Gate
        o_t = tf.nn.sigmoid(tf.matmul(x, tf.transpose(self.W[2])) + tf.matmul(pre_h, tf.transpose(self.U[2])))

        # New memory for new information
        n_c_t = tf.nn.tanh(tf.matmul(x, tf.transpose(self.W[3])) + tf.matmul(pre_h, tf.transpose(self.U[3])))

        # Combination between storing information and new information
        c = tf.multiply(f_t, pre_c) + tf.multiply(i_t, n_c_t)

        # How information are allowed to be output of cell
        h = tf.multiply(o_t, tf.nn.tanh(c))

        return tf.stack([h, c])
         