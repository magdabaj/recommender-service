import json
import sys
from movies_recommender import recommend_movies

import pandas as pd
from flask import Flask, jsonify, request

app = Flask(__name__)


@app.route('/<name>')
def hello_world(name):
    return 'Hello World! {}'.format(name)

@app.route('/recommend/<user_id>', methods=['POST'])
def recommend(user_id):
    ratings = request.json.get('ratings')
    movies = request.json.get('movies')
    movies_df = pd.json_normalize(movies)
    ratings_df = pd.json_normalize(ratings)
    print('movies', file=sys.stderr)
    print(movies_df, file=sys.stderr)
    print('Ratings ', ratings_df)
    recommended_movies = recommend_movies(movies_df, ratings_df, int(user_id))
    print('recommended_movies', recommended_movies)
    return recommended_movies.to_json(orient='records')


if __name__ == '__main__':
    app.run(debug=True)
