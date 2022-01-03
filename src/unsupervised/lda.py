import gensim
from gensim.models import LdaMulticore
import pandas as pd


def lda_all(num_topics=1):
    dictionary = gensim.corpora.Dictionary.load('./data/processed/dictionary')
    train_df = pd.read_csv('./data/processed/train_unsupervised.csv', sep=';', parse_dates=['date'])
    docs = train_df['avis'].str.split()
    bow_corpus = [dictionary.doc2bow(doc) for doc in docs]

    lda_model = LdaMulticore(
        bow_corpus,
        num_topics=num_topics,
        id2word=dictionary,
        passes=20,
        workers=4
    )
    for i in range(num_topics):
        print(f"Topics {i}")
        print(f"  Words: {[t[0] for t in lda_model.show_topic(i)]}")


def lda_split(num_topics_per_split=1):
    train_df = pd.read_csv('./data/processed/train_unsupervised.csv', sep=';', parse_dates=['date'])
    for i in range(5):
        dictionary = gensim.corpora.Dictionary.load(f'./data/processed/dictionary_split_{i}')
        docs = train_df[train_df['note'] == i]['avis'].str.split()
        bow_corpus = [dictionary.doc2bow(doc) for doc in docs]

        lda_model = LdaMulticore(
            bow_corpus,
            num_topics=num_topics_per_split,
            id2word=dictionary,
            passes=20,
            workers=4
        )
        print(f"Note: {i}")
        for j in range(num_topics_per_split):
            print(f"  Topics {j}")
            print(f"    Words: {[t[0] for t in lda_model.show_topic(j)]}")
