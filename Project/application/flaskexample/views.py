from flask import render_template, request
from flaskexample import app
from predictions import generate_alternatives
import gensim

path = "/Users/stevenfelix/Documents/DataScience_local/Insight/"
file = 'model_full_50M_sg0_sz250_win5_min3_hs1_neg0'
model = gensim.models.word2vec.Word2Vec.load(path+file)

@app.route('/')
@app.route('/input')
def query_input():
    return render_template("input.html")

@app.route('/output')
def query_output():
  query = request.args.get('query')
  suggestions = generate_alternatives(query, model, 5, 5)
  return render_template("output.html", suggestions=suggestions, query=query)
    