from flask import render_template, request
from flaskexample import app
import pandas as pd
from predictions import predict_similar
import gensim

path = "/Users/stevenfelix/Documents/DataScience_local/Insight/"
file = 'model_full_50M_1_250_5_3'
model = gensim.models.word2vec.Word2Vec.load(path+file)

@app.route('/')
@app.route('/input')
def query_input():
    return render_template("input.html")

@app.route('/output')
def query_output():
  query = request.args.get('query')
  suggestions=predict_similar(query, model)
  return render_template("output.html", suggestions=suggestions, query=query)
    