import pandas as pd
import gensim


def save_dictionary_split(df):
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
    dictionary = gensim.corpora.Dictionary([doc.split(' ') for doc in df['avis']])
    dictionary.save('./data/processed/dictionary')


train_df = pd.read_csv('./data/processed/train_unsupervised.csv', sep=';', parse_dates=['date'])
save_dictionary_split(train_df)
save_dictionary(train_df)
