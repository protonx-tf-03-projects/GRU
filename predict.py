import os
from argparse import ArgumentParser
import tensorflow as tf
import numpy as np
import json
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--review-sentence", default="The actors are really famous", type=str)
    parser.add_argument("--model-path", default="tmp/model/gru.h5py", type=str)
    parser.add_argument(
        "--vocab-path", default='tmp/saved_vocab/tokenizer.json', type=str)
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
    # Load Tokenizer
    with open(args.vocab_path) as file:
        data = json.load(file)
        tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(data)

    # Tokenize the reviews_sentence
    test_sentence = []
    test_sentence.append(args.review_sentence)
    test_sentence = tokenizer.texts_to_sequences(test_sentence)
    input_1 = tf.keras.preprocessing.sequence.pad_sequences(test_sentence, maxlen=args.max_length,
                                                  padding='post', truncating='post')
    input_1 = input_1.astype('int64')

    # Load model
    model = tf.keras.models.load_model(args.model_path)

    # Predicting
    print('---------------------Prediction Result: -------------------')
    results = model.predict(input_1)

    # Load old label dictionary
    with open('label.json') as f:
        label_dict = json.load(f)
    # Reverse label dictionary
    new_label_dict = {}
    for i in label_dict.keys():
        new_label_dict[label_dict[i]] = i

    # Use new Dictionary for choosing the feature
    index = np.argmax(results)
    print("The Review Sentence is: ", new_label_dict[index])



