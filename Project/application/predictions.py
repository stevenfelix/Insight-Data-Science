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
    
# %%
""" These generate alternative queries and score them and filter them """
def generate_alternatives(query, model, n_syn=5, topn=3, rmv_stop_words=False):
    syns = get_similar(query, n_syn, model, rmv_stop_words) # synonyms
    combs = get_combinations(syns) # combinations
    probs = [model.score([sug])[0] for sug in combs] # probabilities
    preds_probs =[(p,q) for p,q in zip(probs,combs)] # combine with queries
    q_score = model.score([clean(query)])[0] # score for original query
    sd = get_sd(preds_probs)
    preds_1sd = [(x,y) for x,y in preds_probs if np.abs(x-q_score)<=sd] # keep just those within 1 sd
    preds_1sd.sort(reverse=True)
    #print("original query: {}".format(query))
    #print("score: {}".format(q_score))
    #print("sd of all results: {}".format(sd))
    #print("number of results within 1 SD of original query score: {}".format(len(preds_1sd)))
    return clean_preds(preds_1sd, topn)

def get_similar(query, n_syn, model, rmv_stop_words):
    q = clean(query, rmv_stop_words=rmv_stop_words)
    d = [[x] for x in q]
    for x in d:
        x.extend([syn for syn,_ in model.most_similar(x[0],topn=n_syn)])
    return d

def get_combinations(l):
    combs = [x for x in product(*l)]
    return combs

def get_sd(tups):
    vals = [x for x,_ in tups]
    return np.std(vals)

def clean_preds(pred_scores, topn=3):
    """ returns the top n by iterating n times (list is sorted already). and
    joins the list of strings into single string"""
    clean = []
    i = 0
    for _,query in pred_scores:
        i+=1
        if i > topn: break
        clean.append(' '.join(query))    
        #clean.append((score, ' '.join(query)))
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