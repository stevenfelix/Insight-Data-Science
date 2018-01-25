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
from itertools import product
import numpy as np
from bs4 import BeautifulSoup
import requests

tokenizer = RegexpTokenizer(r'\w+') # tokens separated by white spice
stops = set(stopwords.words('english')) # list of english stop words
lemma = WordNetLemmatizer()

# %% Clean

## https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/deepir.ipynb
import re

contractions = re.compile(r"'|-|\"")
# all non alphanumeric
symbols = re.compile(r'(\W+)', re.U)
# single character removal
singles = re.compile(r'(\s\S\s)', re.I|re.U)
# separators (any whitespace)
seps = re.compile(r'\s+')
# tokenizer
tokenizer = RegexpTokenizer(r'\w+') # tokens separated by white spice
# stop words
stops = set(stopwords.words('english')) # list of english stop words

# cleaner (order matters)
def clean(text, rmv_stop_words=True, return_tokens=False): 
    text = text.lower()
    text = contractions.sub('', text)
    text = symbols.sub(r' \1 ', text)
    text = singles.sub(' ', text)
    text = seps.sub(' ', text)
    tokens = tokenizer.tokenize(text)     # tokenize
    if rmv_stop_words:
        tokens = [i for i in tokens if not i in stops] # remove stop words
        text = ' '.join(tokens)
    if return_tokens:
        return tokens
    return text

# sentence splitter
#alteos = re.compile(r'([!\?])')
#def sentences(l):
#    l = alteos.sub(r' \1 .', l).rstrip("(\.)*\n")
#    return l.split(".")

    
# %%
""" These generate alternative queries and score them and filter them """

def generate_alternatives(query, n, model, rmv_stop_words=True, return_tokens=True):
    print('getting similar words')
    syns = get_similar(query, n, model, rmv_stop_words=rmv_stop_words, return_tokens=return_tokens) # synonyms
    print('making combinations')
    combs = get_combinations(syns) # combinations
    # generatings probaiblity scores
    probs = [model.score([sug])[0] for sug in combs] # probabilities
    preds_probs =[(p,q) for p,q in zip(probs,combs)] # combine with queries
    q_score = model.score([tokenizer.tokenize(query)])[0] # score for original query
    sd = get_sd(preds_probs)
    preds_1sd = [(x,y) for x,y in preds_probs if np.abs(x-q_score)<=sd] # keep just those within 1 sd
    preds_1sd.sort(reverse=True)
    #print("original query: {}".format(query))
    #print("score: {}".format(q_score))
    #print("sd of all results: {}".format(sd))
    #print("number of results within 1 SD of original query score: {}".format(len(preds_1sd)))
    return preds_1sd

def get_similar(query, n, model, rmv_stop_words, return_tokens):
    q = clean(query, rmv_stop_words=rmv_stop_words, return_tokens=return_tokens)
    # turn each word  of query into its own list
    d = [[x] for x in q]
    for x in d:
        # for each word in original query, add topn similar words to list
        x.extend([syn for syn,_ in model.most_similar(x[0],topn=n)])
    return d

def get_combinations(l):
    combs = [x for x in product(*l)]
    return combs

def get_sd(tups):
    vals = [x for x,_ in tups]
    return np.std(vals)

def clean_preds(pred_scores, topn=3):
    clean = []
    i = 0
    for score,query in pred_scores:
        i+=1
        if i > topn: break
        clean.append((score, ' '.join(query)))
    return clean

# %%

"""These query stack overflow and return and parse the serach results"""

def get_query_results(query):
    url = 'https://stackoverflow.com/search?q='+'+'.join(query)
    r = requests.get(url)
    soup = BeautifulSoup(r.text, 'lxml')
    return parse_results(soup)
    
def parse_results(soup):
    l = []
    results = soup.find_all("div", class_="question-summary search-result")
    for result in results:
        votes = [v.get_text() for v in result.find_all("strong")]
        link = result.find("div", class_="result-link").find('a')
        query = link.attrs['title']
        url = link.attrs['href']
        votes.extend([query,url])
        l.append(tuple(votes))
    return l