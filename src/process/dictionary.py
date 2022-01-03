import spacy
from spacy.lang.fr import stop_words
from nltk.stem.snowball import SnowballStemmer
import pandas as pd
import json
import gensim

nlp = spacy.load('fr_core_news_sm')
nlp.max_length = 3737815
stemmer = SnowballStemmer(language='french')


def is_valid_word(word):
    return not (word.is_punct or word.is_stop or len(word.text) < 3)


def process_text(text):
    doc = nlp(text)
    words = [stemmer.stem(w.lemma_.lower().strip()) for w in doc if is_valid_word(w)]
    return ' '.join(words)


def save_dictionary_split(df):
    df['avis'] = df['avis'].apply(lambda s: process_text(s))
    avis = list(zip(df['note'], df['avis']))
    avis_dict = {
        0: [],
        1: [],
        2: [],
        3: [],
        4: []
    }
    for (n, a) in avis:
        avis_dict[n].append(a.split(' '))
    dictionary = {k: gensim.corpora.Dictionary(v) for k, v in avis_dict.items()}
    for k, v in dictionary.items():
        v.save(f'./data/processed/dictionary_split_{k}')


def save_dictionary(df):
    df['avis'] = df['avis'].apply(lambda s: process_text(s))
    dictionary = gensim.corpora.Dictionary([doc.split(' ') for doc in df['avis']])
    dictionary.save('./data/processed/dictionary')


train_df = pd.read_csv('./data/clean/train.csv', sep=';', parse_dates=['date'])
save_dictionary_split(train_df)
save_dictionary(train_df)
