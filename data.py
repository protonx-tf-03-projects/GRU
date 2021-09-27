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
    '''Encode labels to categorical'''
    labels.replace(data_classes, inplace=True)

    labels_target = labels.values
    labels_target = tf.keras.utils.to_categorical(labels_target)

    return labels_target
  
  def removeHTML(self, text):
    '''Remove html tags from a string'''
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)

  def sentence_cleaning(self, sentence):
    '''Cleaning text'''
    out_sentence = []
    for sent in tqdm(sentence):
      sent = self.removeHTML(sent)
      text = re.sub("[^a-zA-Z]", " ", sent)
      word = word_tokenize(text.lower())

      lemmatizer = WordNetLemmatizer()

      lemm_word = [lemmatizer.lemmatize(i) for i in word]

      out_sentence.append(lemm_word)
    return (out_sentence)

  def data_processing(self, sentences, labels):
    '''Preprocessing both text and labels'''
    print("|--data_processing ...")
    sentences = self.sentence_cleaning(sentences)
    labels = self.labels_encode(labels, data_classes=self.data_classes)
    
    return sentences, labels

  def build_tokenizer(self, sentences, vocab_size, char_level=False):
    print("|--build_tokenizer ...")
    tokenizer = tf.keras.preprocessing.text.Tokenizer(
        num_words= vocab_size, oov_token=OOV, char_level=char_level)
    tokenizer.fit_on_texts(sentences)

    return tokenizer

  def tokenize(self, tokenizer, sentences, max_length):
    print("|--tokenize ...")
    sentences = tokenizer.texts_to_sequences(sentences)
    sentences = tf.keras.preprocessing.sequence.pad_sequences(sentences, maxlen=max_length,
                                                              padding=PADDING, truncating=TRUNC)
    
    return sentences

  def load_dataset(self, max_length, data_name, label_name):
    print(" ")
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

    X_train, X_val, y_train, y_val = train_test_split(
        sentences, labels, test_size=test_size, stratify=labels, random_state=42)

    # Convert to tensor
    train_dataset = tf.data.Dataset.from_tensor_slices((tf.convert_to_tensor(
        X_train, dtype=tf.int64), tf.convert_to_tensor(y_train, dtype=tf.int64)))
    train_dataset = train_dataset.shuffle(buffer_size).batch(batch_size)

    val_dataset = tf.data.Dataset.from_tensor_slices((tf.convert_to_tensor(
        X_val, dtype=tf.int64), tf.convert_to_tensor(y_val, dtype=tf.int64)))
    val_dataset = val_dataset.shuffle(buffer_size).batch(batch_size)
    
    with open('label.json', 'w') as f:
            json.dump(self.label_dict, f)
   
    return train_dataset, val_dataset
