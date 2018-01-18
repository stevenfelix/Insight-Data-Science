from flask import render_template, request
from flaskexample import app
import pandas as pd
from predictions import predict_similar
import gensim

path = "/Users/stevenfelix/Documents/DataScience_local/Insight/"
file = 'model_full_30M_sg_200_5_2'
model = gensim.models.word2vec.Word2Vec.load(path+file)


@app.route('/')
@app.route('/input')
def query_input():
    return render_template("input.html")

@app.route('/output')
def query_output():
  # pull 'query' from input field and store it
  query = request.args.get('query')
  # just select the Cesareans  from the birth dtabase for the month that the user inputs
  print(query)
  suggestions=predict_similar(query, model)
  print(suggestions)
  #births = []
  #for i in range(0,query_results.shape[0]):
  #    births.append(dict(index=query_results.iloc[i]['index'], attendant=query_results.iloc[i]['attendant'], birth_month=query_results.iloc[i]['birth_month']))
  #    the_result = ModelIt(patient,births)
  return render_template("output.html", suggestions=suggestions.to_html(), query=query)
    