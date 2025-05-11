import os
import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.models import load_model
import numpy as np
from train_genre_classifier import preprocess, train_model

# Constants
BOOKS_FOLDER = "books_data"
TOP_K = 5

# Function definitions

def load_jsonl_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]

def build_lookups():
    author_data = load_jsonl_file(os.path.join(BOOKS_FOLDER, "goodreads_book_authors.json"))
    authors = {a['author_id']: a for a in author_data}

    genre_data = load_jsonl_file(os.path.join(BOOKS_FOLDER, "goodreads_book_genres_initial.json"))
    genres = {g['book_id']: list(g['genres'].keys()) for g in genre_data}

    series_data = load_jsonl_file(os.path.join(BOOKS_FOLDER, "goodreads_book_series.json"))
    series_map = {s['series_id']: s['title'] for s in series_data}

    works_data = load_jsonl_file(os.path.join(BOOKS_FOLDER, "goodreads_book_works.json"))
    works = {w['best_book_id']: w for w in works_data}

    return authors, genres, series_map, works

def load_books():
    books_path = os.path.join(BOOKS_FOLDER, "goodreads_books.json")
    with open(books_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]

def preprocess_books(books, authors, genres, series_map, works):
    processed = []
    for book in books:
        title = book.get("title", "").strip()
        book_id = book.get("book_id")
        desc = book.get("description", "").strip()
        shelf_tags = " ".join([shelf["name"] for shelf in book.get("popular_shelves", [])])

        # add in metrics with author
        author_id = book.get("authors", [{}])[0].get("author_id")
        author = authors.get(author_id, {})
        author_name = author.get("name", "")

        # add in metrics with genres
        genre_list = genres.get(book_id, [])

        # add in metrics with series
        series_info = book.get("series", [])
        series_names = [series_map.get(sid) for sid in series_info if sid in series_map]

        # add in metrics with works
        work = works.get(book_id, {})
        rating_dist = work.get("rating_dist", "")
        reviews_count = work.get("reviews_count", 0)

        combined_text = f"{desc} {shelf_tags} {' '.join(genre_list)} {' '.join(series_names)} by {author_name}"

        if title and combined_text:
            processed.append({
                "book_id": book_id,
                "title": title,
                "text": combined_text,
                "author": author_name,
                "genres": genre_list,
                "series": series_names,
                "rating": float(book.get("average_rating", 0)),
                "reviews_count": int(reviews_count)
            })
    return pd.DataFrame(processed)

def predict_genres(text, model, tfidf, mlb):
    vec = tfidf.transform([text]).toarray()
    pred = model.predict(vec)
    pred_labels = mlb.inverse_transform((pred > 0.5).astype(int))
    return pred_labels[0] if pred_labels else []

def build_similarity_matrix(df):
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf_matrix = tfidf.fit_transform(df["text"])
    return cosine_similarity(tfidf_matrix)

def recommend_books(title, df, sim_matrix, top_k=TOP_K):
    if title not in df["title"].values:
        print(f"Book titled '{title}' not found.")
        return []
    idx = df[df["title"] == title].index[0]
    sim_scores = list(enumerate(sim_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_indices = [i for i, _ in sim_scores[1:top_k + 1]]
    return df.iloc[sim_indices][["title", "author", "rating"]]

def main():
    print("Loading auxiliary data...")
    authors, genres, series_map, works = build_lookups()
    print("Loading main books file...")
    books = load_books()
    print("Processing data...")
    df_books = preprocess_books(books, authors, genres, series_map, works)
    print(f"Books loaded: {len(df_books)}")

    print("Building similarity matrix...")
    sim_matrix = build_similarity_matrix(df_books)

    query_title = "Good Harbor"
    recommendations = recommend_books(query_title, df_books, sim_matrix)

    print(f"\nBooks similar to '{query_title}':")
    for _, row in recommendations.iterrows():
        print(f"- {row['title']} by {row['author']} (Rating: {row['rating']})")

    print("\nLoading genre classifier model...")
    from train_genre_classifier import preprocess as genre_preprocess, train_model

    genre_df = genre_preprocess()
    model_path = "genre_model.h5"

    if os.path.exists(model_path):
        genre_model = load_model(model_path)
        genre_tfidf, genre_mlb = train_model(genre_df)
        print("Model and vectorizer loaded.")

        # Predict genres for the first book
        sample_text = df_books.iloc[0]['text']
        predicted_genres = predict_genres(sample_text, genre_model, genre_tfidf, genre_mlb)
        print(f"\nPredicted genres for '{df_books.iloc[0]['title']}': {predicted_genres}")
    else:
        print("Trained genre model not found. Run train_genre_classifier.py first.")


if __name__ == "__main__":
    main()
