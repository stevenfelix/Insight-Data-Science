#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 15:42:56 2018

@author: stevenfelix
"""


# Logging code taken from http://rare-technologies.com/word2vec-tutorial/
# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

from nltk.tokenize import RegexpTokenizer # tokenizing
from nltk.corpus import stopwords  # list of stop words
from nltk.stem.wordnet import WordNetLemmatizer # lemmatizer
import pandas as pd

tokenizer = RegexpTokenizer(r'\w+') # tokens separated by white spice
stops = set(stopwords.words('english')) # list of english stop words
lemma = WordNetLemmatizer()

def clean(title, rmv_stop_words=False):
    tokens = tokenizer.tokenize(title.lower())     # tokenize
    if rmv_stop_words:
        tokens = [i for i in tokens if not i in stops] # remove stop words
    normalized = [lemma.lemmatize(token) for token in tokens] # lemma
    return normalized

def predict_similar(query, model, rmv_stop_words=False):
    l = []
    q = clean(query, rmv_stop_words=rmv_stop_words)
    
    print('Original query: {}'.format(query))

    for word in q:
        missing = q[:]
        ind = q.index(word)
        missing.remove(word)
        print('\n')
        for syn in model.most_similar([word],topn=3):
            full = missing[:]
            full.insert(ind,syn[0])
            l.append(' '.join(full))
    if l is None:
        return "function did not return results"
    else:
        return l
    