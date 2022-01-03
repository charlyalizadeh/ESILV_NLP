import spacy
from nltk.stem.snowball import SnowballStemmer
import pandas as pd


nlp = spacy.load('fr_core_news_sm')
nlp.max_length = 3737815
stemmer = SnowballStemmer(language='french')


def is_valid_word(word):
    return not (word.is_punct or word.is_stop or len(word.text) < 3)


def process_text(text):
    doc = nlp(text)
    words = [stemmer.stem(w.lemma_.lower().strip()) for w in doc if is_valid_word(w)]
    words = [w for w in words if ';' not in w]
    return ' '.join(words)


train_df = pd.read_csv('./data/clean/train.csv', sep=';', parse_dates=['date'])
train_df['avis'] = train_df['avis'].apply(lambda s: process_text(s))
train_df.replace('', float('NaN'), inplace=True)
train_df.dropna(subset=['avis'], inplace=True)
train_df.to_csv('./data/processed/train_unsupervised.csv', index=False, sep=';')
