import pandas as pd
import torch
import spacy
from transformers import FlaubertModel, FlaubertTokenizer

#import fasttext.util
#fasttext.util.download_model('fr', if_exists='ignore')
#ft = fasttext.load_model('cc.fr.300.bin')

#train = pd.read_csv('../data/clean/train.csv', sep=';')
#test = pd.read_csv('../data/clean/test.csv', sep=';')
#nlp = spacy.load("fr_core_news_sm")
modelname = 'flaubert/flaubert_base_cased'
#
#flaubert, log = FlaubertModel.from_pretrained(modelname, output_loading_info=True)
flaubert_tokenizer = FlaubertTokenizer.from_pretrained(modelname, do_lowercase=False)
#
#def process_df(df):
#    # Drop useless columns
#    df.drop(['auteur'], axis=1, inplace=True)
#
#    # Convert categorical features to numerical
#    for c in ['assureur', 'produit']:
#        df[c] = df[c].astype('category')
#        df[c] = df[c].cat.codes
#
#
sentence = "Salut ça va ?"
sentence_2 = "salut ça va ? Oui et toi ?"
token_ids = torch.tensor([flaubert_tokenizer.encode(sentence)])
token_ids_2 = torch.tensor([flaubert_tokenizer.encode(sentence_2)])
print(token_ids)
print(token_ids.shape)
print(token_ids_2)
print(token_ids_2.shape)
#last_layer = flaubert(token_ids)[0]
#print(last_layer.shape)
