# -*- coding: utf-8 -*-
"""
Created on Wed Aug 18 15:45:06 2021
@author: joeeislovely
"""

## Code for automated data processing for GRU problem
## Link: https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
## Author: Sang M. Nguyen
## Date: 2021-08-18

# Libraries imported.
import pandas as pd
from bs4 import BeautifulSoup

import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

from sklearn.model_selection import train_test_split

# Class for processing data.
class Dataset:
    def __init__(self, data):
        
        """
        Reads the input data, in this example, the data is adopted from Kaggle.
        """
        
        self.df = pd.read_csv(data)
        
    def _removeHTMLRegex(self):
        
        """
        Processes the raw input data by removing HTML and Regex.
        Also it lowers all the chars for better training.
        """
        
        soup = BeautifulSoup(self.text)
        self.text = soup.get_text().lower().replace(r'[^a-zA-Z0-9]',' ')
        
    def _stemming(self):
        
        """
        Stems the input text.
        """
        
        sw = set(stopwords.words('english'))
        snow = SnowballStemmer('english')
        self.text=' '.join([snow.stem(word) for word in self.text.split() if word not in sw])
        
    def _cleanData(self):
        
        """
        Applies above functions for the data.
        Also encodes the sentiment to 0 and 1. 
        """
        
        self.df.review = self.df.review.apply(_removeHTMLRegex).apply(_stemming)
        self.df.sentiment.replace({'negative':0, 'positive':1}, inplace=True)
    
    def _splitData(self):
        
        """
        Splits the data into train and test set.
        """
        
        self.train, self.test = train_test_split(self.df, test_size=0.3)
        get_samples = lambda df : (df["review"].values, df["sentiment"].values)
        self.train_data, self.train_labels = get_samples(self.train)
        self.val_data, self.val_labels = get_samples(self.test) 