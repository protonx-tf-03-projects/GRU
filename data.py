# Libraries imported.
import re
import tensorflow as tf
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from constant import *

nltk.download('punkt')
nltk.download('wordnet')

class Dataset:
  def __init__(self, data_path, vocab_size, data_classes):
    self.data_path = data_path
    self.vocab_size = vocab_size
    self.data_classes = data_classes
    self.sentences_tokenizer = None

  def labels_encode(self, labels, data_classes):
    # data_classes = {'negative': 0, 'positive': 1}

    labels.replace(data_classes, inplace=True)

    labels_target = labels.values
    labels_target = tf.keras.utils.to_categorical(labels_target)

    return labels_target
    
  def sentence_cleaning(self, sentence):
    out_sentence = []
    for sent in tqdm(sentence):
      text = re.sub("[^a-zA-Z]", " ", sent)
      word = word_tokenize(text.lower())

      lemmatizer = WordNetLemmatizer()

      lemm_word = [lemmatizer.lemmatize(i) for i in word]

      out_sentence.append(lemm_word)
    return (out_sentence)

  def data_processing(self, sentences, labels):

    sentences = self.sentence_cleaning(sentences)
    labels = self.labels_encode(labels, data_classes=self.data_classes)
    
    print("===data_processing===")
    return sentences, labels

  def build_tokenizer(self, sentences, vocab_size, char_level=False):
    tokenizer = tf.keras.preprocessing.text.Tokenizer(
        num_words= vocab_size, oov_token=OOV, char_level=char_level)
    tokenizer.fit_on_texts(sentences)

    print("==build_tokenizer==")
    return tokenizer

  def tokenize(self, tokenizer, sentences, max_length):
    sentences = tokenizer.texts_to_sequences(sentences)
    sentences = tf.keras.preprocessing.sequence.pad_sequences(sentences, maxlen=max_length,
                                                              padding=PADDING, truncating=TRUNC)
    
    print("==tokenize==")
    return sentences

  def load_dataset(self, max_length, data_name, label_name):
    print("Load dataset ... ")
    datastore = pd.read_csv(self.data_path)
    sentences = datastore[data_name]
    labels = datastore[label_name]

    # Cleaning
    sentences, labels = self.data_processing(sentences, labels)
    
    # Tokenizing
    self.sentences_tokenizer = self.build_tokenizer(sentences, self.vocab_size)
    tensor = self.tokenize(
        self.sentences_tokenizer, sentences, max_length)
    
    print("Done! Next to ... ")
    print(" ")
    return tensor, labels
                                                                  
  def build_dataset(self, max_length=128, test_size=0.2, buffer_size=128, batch_size=128, data_name='review', label_name='sentiment'):
    sentences, labels = self.load_dataset(
        max_length, data_name, label_name)

    X_train, X_test, y_train, y_test = train_test_split(
        sentences, labels, test_size=test_size, stratify=labels, random_state=42)

    # Convert to tensor
    train_dataset = tf.data.Dataset.from_tensor_slices((tf.convert_to_tensor(
        X_train, dtype=tf.int64), tf.convert_to_tensor(y_train, dtype=tf.int64)))
    train_dataset = train_dataset.shuffle(buffer_size).batch(batch_size)

    val_dataset = tf.data.Dataset.from_tensor_slices((tf.convert_to_tensor(
        X_test, dtype=tf.int64), tf.convert_to_tensor(y_test, dtype=tf.int64)))
    val_dataset = val_dataset.shuffle(buffer_size).batch(batch_size)

    return train_dataset, val_dataset
