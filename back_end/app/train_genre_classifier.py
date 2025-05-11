# train_genre_classifier.py

import os
import json
import pickle
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model

BOOKS_FOLDER = "books_data"
MODEL_PATH = "models/genre_model.h5"
TFIDF_PATH = "models/tfidf_vectorizer.pkl"
MLB_PATH = "models/label_binarizer.pkl"
MAX_SAMPLES = 50000  # Limit to avoid memory errors


def load_books():
    with open(os.path.join(BOOKS_FOLDER, "goodreads_books.json"), 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def load_genres():
    path = os.path.join(BOOKS_FOLDER, "goodreads_book_genres_initial.json")
    with open(path, 'r', encoding='utf-8') as f:
        return {json.loads(line)['book_id']: list(json.loads(line)['genres'].keys()) for line in f if line.strip()}


def preprocess():
    books = load_books()
    genres = load_genres()

    data = []
    for book in books:
        book_id = book.get("book_id")
        if not book_id or book_id not in genres:
            continue

        title = book.get("title", "").strip()
        desc = book.get("description", "").strip()
        shelves = " ".join([s["name"] for s in book.get("popular_shelves", [])])
        text = f"{title} {desc} {shelves}"

        if text:
            data.append({
                "book_id": book_id,
                "text": text,
                "genres": genres[book_id]
            })

        if len(data) >= MAX_SAMPLES:
            break

    return pd.DataFrame(data)


def save_pickle(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def train_model(df):
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
    X = tfidf.fit_transform(df['text']).toarray()

    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(df['genres'])

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    model = Sequential([
        Dense(512, activation='relu', input_shape=(X.shape[1],)),
        Dropout(0.3),
        Dense(256, activation='relu'),
        Dropout(0.3),
        Dense(y.shape[1], activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_val, y_val), callbacks=[early_stop])

    model.save(MODEL_PATH)
    save_pickle(tfidf, TFIDF_PATH)
    save_pickle(mlb, MLB_PATH)

    print(f"Model saved to {MODEL_PATH}")
    print(f"TF-IDF vectorizer saved to {TFIDF_PATH}")
    print(f"Label binarizer saved to {MLB_PATH}")

    return tfidf, mlb


if __name__ == '__main__':
    df = preprocess()
    tfidf, mlb = train_model(df)
