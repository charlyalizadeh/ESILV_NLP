# French pre trained word embedding
## * https://fasttext.cc/docs/en/crawl-vectors.html

# Or tokenizer
## FlauBERT
## RoBERT

import spacy
from gensim.models import KeyedVectors
from gensim.models.fasttext import load_facebook_vectors
import numpy as np


class Embedder:
    def __init__(self, embedder_type, spacy_embedding='fr_core_news_md', word2vec_embedding='no_pre_process', spacy_nlp='fr_core_news_md'):
        self.embedder_type = embedder_type
        self.nlp = spacy.load(spacy_nlp)
        #self.embedder = None
        if embedder_type == 'spacy':
            self.embedder = spacy.load(spacy_embedding)
            self.embedding_size = 300
        if embedder_type == 'fasttext':
            try:
                self.embedder = KeyedVectors.load('models/fasttext.model')
            except FileNotFoundError:
                self.embedder = load_facebook_vectors('data/word_embedding/cc.fr.300.bin')
                self.embedder.save('models/fasttext.model')
            self.embedding_size = 300
        if embedder_type == 'word2vec':
            raise NotImplementedError
            #try:
            #    self.embedder = KeyedVectors.load(f'models/word2vec_{word2vec_embedding}.model')
            #except FileNotFoundError:
            #    self.embedder = KeyedVectors.load_word2vec_format(f'data/word_embedding/word2vec_{word2vec_embedding}.bin', binary=True)
            #    self.embedder.save(f'models/word2vec_{word2vec_embedding}.model')
            #self.embedding_size = 500

    def __getitem__(self, key):
        if self.embedder_type == 'spacy':
            return self.embedder(key).vector
        else:
            vect = np.zeros(self.embedding_size)
            doc = self.nlp(key)
            nb_doc = len(doc)
            for d in doc:
                try:
                    vect += self.embedder[d.text]
                except Exception:
                    nb_doc -= 1
            return vect / nb_doc
