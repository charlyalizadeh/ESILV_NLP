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

train_df = pd.read_csv('../data/raw/train.csv', sep=';')
test_df = pd.read_csv('../data/raw/test.csv', sep=';')
for k, v in month_dict.items():
    train_df['date'] = train_df['date'].str.replace(k, v)
    test_df['date'] = test_df['date'].str.replace(k, v)

train_df['date'] = train_df['date'].apply(lambda s: s.strip()[:10])
test_df['date'] = test_df['date'].apply(lambda s: s.strip()[:10])
train_df['date'] = pd.to_datetime(train_df['date'], format='%d %m %Y')
test_df['date'] = pd.to_datetime(test_df['date'], format='%d %m %Y')
os.mkdir('../data/clean', exist_ok=True)
train_df.to_csv('../data/clean/train.csv', sep=';', index=False)
test_df.to_csv('../data/clean/test.csv', sep=';', index=False)
