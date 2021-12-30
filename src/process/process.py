from embedder import Embedder
import pandas as pd


embedder = Embedder('spacy')


def process_df(df):
    df['avis'] = df['avis'].apply(lambda s: ','.join(map(str, embedder[s])))
    for c in ['assureur', 'produit']:
        df[c] = df[c].astype('category')
        df[c] = df[c].cat.codes
    df['dayofweek'] = df['date'].dt.dayofweek
    df['day'] = df['date'].dt.day
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df.drop(['auteur', 'date'], axis=1, inplace=True, errors='ignore')


train_df = pd.read_csv('./data/clean/train.csv', sep=';', parse_dates=['date'])
test_df = pd.read_csv('./data/clean/test.csv', sep=';', parse_dates=['date'])
process_df(train_df)
process_df(test_df)
train_df.to_csv('./data/processed/train.csv', sep=';', index=False)
test_df.to_csv('./data/processed/test.csv', sep=';', index=False)
