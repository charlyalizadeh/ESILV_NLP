from embedder import Embedder
import pandas as pd
import click


def process_df(df, embedder, vectorize=True):
    if vectorize:
        df['avis'] = df['avis'].apply(lambda s: ','.join(map(str, embedder[s])))
    for c in ['assureur', 'produit']:
        df[c] = df[c].astype('category')
        df[c] = df[c].cat.codes
    df['dayofweek'] = df['date'].dt.dayofweek
    df['day'] = df['date'].dt.day
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df.drop(['auteur', 'date'], axis=1, inplace=True, errors='ignore')


@click.argument('embedder_type', default='spacy')
@click.option('--spacy_nlp', default='fr_core_news_md')
@click.option('--spacy_embedding', default='fr_core_news_md')
@click.option('--word2vec_embedding', default='no_pre_process')
@click.option('--vectorize', default=True, type=bool)
@click.command()
def process_supervised(embedder_type, spacy_nlp, spacy_embedding, word2vec_embedding, vectorize):
    train_df = pd.read_csv('./data/clean/train.csv', sep=';', parse_dates=['date'])
    test_df = pd.read_csv('./data/clean/test.csv', sep=';', parse_dates=['date'])
    if vectorize:
        embedder = Embedder(embedder_type, spacy_nlp=spacy_nlp, spacy_embedding=spacy_embedding, word2vec_embedding=word2vec_embedding)
        process_df(train_df, embedder, vectorize)
        process_df(test_df, embedder, vectorize)
        train_df.to_csv(f'./data/processed/train_{spacy_nlp}_{spacy_embedding}.csv', sep=';', index=False)
        test_df.to_csv(f'./data/processed/test_{spacy_nlp}_{spacy_embedding}.csv', sep=';', index=False)
    else:
        process_df(train_df, None, False)
        process_df(test_df, None, False)
        train_df = train_df[['avis', 'note']]
        test_df = test_df[['avis']]
        train_df.to_csv('./data/processed/train_no_embedding.csv', sep=';', index=False)
        test_df.to_csv('./data/processed/test_no_embedding.csv', sep=';', index=False)


if __name__ == '__main__':
    process_supervised()
