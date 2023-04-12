import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MultiLabelBinarizer
import matplotlib.pyplot as plt

from flask import jsonify
from flask import Flask, request
from flask_cors import CORS, cross_origin

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

def recommend_books(user_genres, n_recommendations=5):
    dataset = pd.read_csv("demo.csv")

    genres = dataset["genres"]
    genre_lists = [genre.split(",") for genre in genres]

    mlb = MultiLabelBinarizer()
    genre_encoding = mlb.fit_transform(genre_lists)

    genre_df = pd.DataFrame(genre_encoding, columns=mlb.classes_)

    dataset_encoded = pd.concat([dataset, genre_df], axis=1)

    dataset_encoded = dataset_encoded.drop("genres", axis=1)

    knn_model = NearestNeighbors(n_neighbors=n_recommendations, algorithm="auto")
    knn_model.fit(dataset_encoded.iloc[:, 3:]) 

    user_genres_encoded = mlb.transform([user_genres])

    neighbor_indices = knn_model.kneighbors(user_genres_encoded)[1][0]

    recommended_books = dataset.iloc[neighbor_indices][["book_id", "title", "genres"]]
    return recommended_books




@app.route("/recommend", methods = ['POST'])
def recommend():
    data = request.get_json()
    genres = data["genres"]
    num = data["num"]
    # genres = ["mystery", "adventure", "fantasy"]

    result = recommend_books(genres, num)
    return jsonify(result.to_dict())
#curl -X POST http://localhost:5000/recommend -H "Content-Type: application/json" -d '{"genres":["Travel Literature", "Fanatasy"], "num": 3}'

if __name__ == '__main__':
    app.run()
