import tensorflow as tf
import pandas as pd
import numpy as np
from layers.LSTM import Scratch_LSTM

class LSTM_RNN(tf.keras.Model):
    """
        Using Scratch_LTSTM and some of Dense class for training model
    """

    def __init__(self, units, embedding_size, vocab_size, input_length):
        super(LSTM_RNN,self).__init__()
        self.input_length = input_length
        self.units = units

        self.embedding = tf.keras.layers.Embedding(
            vocab_size,
            embedding_size,
            input_length=input_length
        )

        self.Scratch_LSTM = Scratch_LSTM(units, embedding_size)

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
        pre_layer = tf.stack([
            tf.zeros([batch_size, self.units]),
            tf.zeros([batch_size, self.units])
        ])

        # Put sentence into Embedding
        embedded_sentence = self.embedding(sentence)

        # Use LSTM with every single word in sentence
        for i in range (self.input_length):
            word = embedded_sentence[:, i, :]
            pre_layer = self.Scratch_LSTM(pre_layer, word)

        # Take the last hidden _state
        h, _ = tf.unstack(pre_layer)

        # Using last hidden_state for for predicting or other processing
        return self.classfication_model(h)