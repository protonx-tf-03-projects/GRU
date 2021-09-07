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


class Tanh_RNN(tf.keras.Model):
    """
        Using Tanh and Dense layers for training model
    """

    def __init__(self, units, embedding_size, vocab_size, input_length, num_class):
        super(Tanh_RNN,self).__init__()
        self.input_length = input_length
        self.units = units

        self.embedding = tf.keras.layers.Embedding(
            vocab_size,
            embedding_size,
            input_length=input_length
        )

        self.model = Tanh(units, embedding_size)

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
        pre_layer = tf.zeros([batch_size, self.units])

        # Put sentence into Embedding
        embedded_sentence = self.embedding(sentence)

        # Use Tanh with every single word in sentence
        for i in range(self.input_length):
            word = embedded_sentence[:, i, :]
            pre_layer = self.model(pre_layer, word)

        # Using last hidden_state for for predicting or other processing
        return self.classification_layer(pre_layer)
