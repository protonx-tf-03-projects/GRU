import tensorflow as tf

class LSTM(tf.keras.layers.Layer):
    """
        Scratch LSTM with the equations by modifying the original LSTM tensorflow model
    """
    def __init__(self, units, inp_shape):
        super(LSTM, self).__init__()
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
         

class LSTM_RNN(tf.keras.Model):
    """
        Using LSTM cell and Dense layers for training model
    """

    def __init__(self, units, embedding_size, vocab_size, input_length, num_class):
        super(LSTM_RNN,self).__init__()
        self.input_length = input_length
        self.units = units

        self.embedding = tf.keras.layers.Embedding(
            vocab_size,
            embedding_size,
            input_length=input_length
        )

        self.LSTM = LSTM(units, embedding_size)

        self.classification_layer = tf.keras.models.Sequential([
            tf.keras.layers.Dense(32, input_shape=(units,), activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(num_class, activation='softmax')
        ])
        
    def call(self, sentence):
        """
        param: sentence need to trained
            type: Tensor
            shape: ( batch_size, input_length)

        return: Output predicted by the model
            type: Tensor
            shape: (batch_size,1)
        """
        batch_size = tf.shape(sentence)[0]

        # create hidden_state and context_state
        pre_layer = tf.stack([
            tf.zeros([batch_size, self.units]),
            tf.zeros([batch_size, self.units])
        ])

        # Put sentence into Embedding
        embedded_sentence = self.embedding(sentence)

        # Use LSTM with every single word in sentence
        for i in range (self.input_length):
            word = embedded_sentence[:, i, :]
            pre_layer = self.LSTM(pre_layer, word)

        # Take the last hidden _state
        h, _ = tf.unstack(pre_layer)

        # Using last hidden_state for for predicting or other processing
        return self.classification_layer(h)
