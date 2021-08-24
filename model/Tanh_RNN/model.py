import tensorflow as tf
import pandas as pd
import numpy as np
from layers.LSTM import Tanh

class Tanh_RNN(tf.keras.Model):
    """
        Using Tanh and some of Dense class for training model
    """

    def __init__(self, units, embedding_size, vocab_size, input_length):
        super(Tanh_RNN,self).__init__()
        self.input_length = input_length
        self.units = units

        self.embedding = tf.keras.layers.Embedding(
            vocab_size,
            embedding_size,
            input_length=input_length
        )

        self.model = Tanh(units, embedding_size)

        self.classfication_model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(128, input_shape=(units,), activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
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
        return self.classfication_model(pre_layer)