# app/recommender/collaborative.py

import os
import json
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Path to ratings file
RATINGS_PATH = os.path.join(os.path.dirname(__file__), '..', 'books_data', 'ratings_cleaned.json')

def load_ratings_matrix():
    with open(RATINGS_PATH, encoding='utf-8') as f:
        records = [json.loads(line) for line in f if line.strip()]
    df = pd.DataFrame(records)
    # Ensure correct types
    df['rating'] = df['rating'].astype(float)
    df['user_id'] = df['user_id'].astype(str)
    df['book_id'] = df['book_id'].astype(str)

    # Build user-item matrix
    matrix = df.pivot_table(index='user_id', columns='book_id', values='rating')
    return matrix

def predict_for_user(user_id: str, top_n: int = 5):
    ratings_matrix = load_ratings_matrix()

    if user_id not in ratings_matrix.index:
        return []

    # Compute cosine similarity between users
    similarity = cosine_similarity(ratings_matrix.fillna(0))
    similarity_df = pd.DataFrame(similarity, index=ratings_matrix.index, columns=ratings_matrix.index)

    # Get similar users (excluding self)
    sim_scores = similarity_df[user_id].drop(user_id)
    top_users = sim_scores[sim_scores > 0]

    if top_users.empty:
        return []

    # Weighted average of ratings from similar users
    weighted_ratings = ratings_matrix.loc[top_users.index].T.dot(top_users)
    normalization = top_users.sum()
    predictions = weighted_ratings / normalization

    # Exclude books already rated by the user
    already_rated = ratings_matrix.loc[user_id][ratings_matrix.loc[user_id].notna()].index
    predictions = predictions.drop(index=already_rated, errors="ignore")

    return predictions.sort_values(ascending=False).head(top_n).to_dict()

def collaborative_filtering_scores(user_id: int, top_n: int = 10):
    predictions = predict_for_user(user_id, top_n=top_n)
    return predictions
