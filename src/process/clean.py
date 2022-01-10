import pandas as pd
import os


month_dict = {
        'janvier': '01',
        'février': '02',
        'mars': '03',
        'avril': '04',
        'mai': '05',
        'juin': '06',
        'juillet': '07',
        'août': '08',
        'septembre': '09',
        'octobre': '10',
        'novembre': '11',
        'décembre': '12'
}
os.makedirs('./data/clean', exist_ok=True)


def clean_df(df):
    for k, v in month_dict.items():
        df['date'] = df['date'].str.replace(k, v)
    df['date'] = df['date'].apply(lambda s: s.strip()[:10])
    df['date'] = pd.to_datetime(df['date'], format='%d %m %Y')
    df.dropna(subset=['avis'], inplace=True)
    df.fillna({'avis': ''})
    if 'note' in df.columns:
        df['note'] = df['note'].apply(lambda s: s - 1)


train_df = pd.read_csv('./data/raw/train.csv', sep=';')
test_df = pd.read_csv('./data/raw/test.csv', sep=';')
clean_df(train_df)
clean_df(test_df)
train_df.to_csv('./data/clean/train.csv', sep=';', index=False)
test_df.to_csv('./data/clean/test.csv', sep=';', index=False)
