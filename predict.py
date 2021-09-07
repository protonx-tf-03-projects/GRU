import os
from argparse import ArgumentParser
import tensorflow as tf
from model.gru_rnn import GRU_RNN
from data import Dataset
import pandas as pd
import numpy as np

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--review-sentence", default= "The actors are really famous", type=str)
    parser.add_argument("--model-path", default="tmp/model/gru.h5py", type=str)

    parser.add_argument(
        "--data-classes", default={'negative': 0, 'positive': 1}, type=set)
    parser.add_argument("--data-path", default='data/IMDB_Dataset.csv', type=str)
    parser.add_argument("--data-name", default='review', type=str)

    parser.add_argument("--vocab-size", default=10000, type=int)
    parser.add_argument("--max-length", default=256, type=int)

    # FIXME
    args = parser.parse_args()

    # FIXME
    # Project Description

    print(' ')
    print('---------------------Welcome to GRU Team | TF03 | ProtonX-------------------')
    print('Github: joeeislovely | anhdungpro97 | ttduongtran')
    print('Email: nguyenminh.sangatpa@gmail.com | anhdung1951997@gmail.com | ttduongtran@gmail.com')
    print('---------------------------------------------------------------------')
    print(f'Predicting model with hyper-params:')
    print('===========================')

    # print arguments
    for i, arg in enumerate(vars(args)):
      print('{}. {}: {}'.format(i, arg, vars(args)[arg]))
    print('===========================')

    # FIXME
    # Create Tokenizer
    datastore = pd.read_csv(args.data_path)
    sentences = datastore[args.data_name]

    dataset = Dataset(args.data_path, args.vocab_size, data_classes=args.data_classes)
    tokenizer = dataset.build_tokenizer(sentences, args.vocab_size, char_level=False)

    # Tokenize the reviews_sentence
    input_1 = dataset.tokenize(tokenizer, args.review_sentence, args.max_length)
    input_1 = input_1.astype('int64')
    
    # Load model
    model = tf.keras.models.load_model(args.model_path)

    #Predicting
    print('---------------------Prediction Result: -------------------')
    results = model.predict(input_1)    
    print('Output: {}'.format(results))

