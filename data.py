import pandas as pd
from bs4 import BeautifulSoup

import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

class Dataset:
    def __init__(self, data):
        self.df = pd.read_csv(data)

    def _removeHTMLRegex(text) -> str:
        soup = BeautifulSoup(text)
        return soup.get_text().lower().replace(r'[^a-zA-Z0-9]',' ')
    
    def _stemming(text) -> str:
        sw = set(stopwords.words('english'))
        snow = SnowballStemmer('english')
        text=' '.join([snow.stem(word) for word in text.split() if word not in sw])
        return text
    
    def _cleanData(self, data):
        self.df.review = self.df.review.apply(_removeHTMLRegex).apply(_stemming)
        self.df.sentiment.replace({'negative':0, 'positive':1}, inplace=True)

### TODO: Train test split + tokenizer 