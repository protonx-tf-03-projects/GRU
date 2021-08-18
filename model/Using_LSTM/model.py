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

        # TODO: Update later
        
        return None