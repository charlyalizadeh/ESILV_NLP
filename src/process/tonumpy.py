import numpy as np
import pandas as pd


def get_numpy(df):
    df[[f'avis_{i}' for i in range(300)]] = df['avis'].str.split(pat=',', expand=True).astype(float)
    df.drop('avis', axis=1, inplace=True)
    return df.to_numpy()


train_df = pd.read_csv('./data/processed/train.csv', sep=';')
test_df = pd.read_csv('./data/processed/test.csv', sep=';')
train_np = get_numpy(train_df)
test_np = get_numpy(test_df)
np.save('./data/processed/train.npy', train_np)
np.save('./data/processed/test.npy', test_np)
