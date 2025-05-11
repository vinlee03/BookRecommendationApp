# books_data/combine_books.py

import os
import json
import pandas as pd
from tqdm import tqdm

# Setup paths
BASE_DIR = os.path.dirname(__file__)
GR_PATH = os.path.join(BASE_DIR, 'goodreads')
KG_PATH = os.path.join(BASE_DIR, 'kaggle_books.csv')
OUTPUT_PATH = os.path.join(BASE_DIR, 'books_merged.csv')

# Load kaggle_books.csv
kaggle_df = pd.read_csv(KG_PATH, dtype=str).fillna("")

# Normalize ISBN
kaggle_df['isbn13'] = kaggle_df['isbn13'].str.replace("-", "").str.strip()
kaggle_df['title'] = kaggle_df['title'].str.strip()

# Read goodreads_books.json
gr_books = []
with open(os.path.join(GR_PATH, 'goodreads_books.json'), encoding='utf-8') as f:
    for line in f:
        gr_books.append(json.loads(line))
gr_books_df = pd.DataFrame(gr_books)

# Extract author data
author_map = {}
with open(os.path.join(GR_PATH, 'goodreads_book_authors.json'), encoding='utf-8') as f:
    for line in f:
        item = json.loads(line)
        author_map[item['author_id']] = item['name']

# Map author names
def get_author_name(author_entries):
    if not author_entries:
        return ""
    return ", ".join([author_map.get(author['author_id'], "") for author in author_entries if 'author_id' in author])

gr_books_df['authors'] = gr_books_df['authors'].apply(get_author_name)

# Extract genres
genre_map = {}
with open(os.path.join(GR_PATH, 'goodreads_book_genres_initial.json'), encoding='utf-8') as f:
    for line in f:
        item = json.loads(line)
        genre_map[item['book_id']] = list(item['genres'].keys())[0] if item['genres'] else ""

gr_books_df['genres'] = gr_books_df['book_id'].map(genre_map)

# Basic fields
gr_books_df = gr_books_df[[
    'book_id', 'title', 'authors', 'description',
    'average_rating', 'ratings_count', 'genres', 'isbn13'
]].fillna("")

# Merge
combined_df = pd.merge(
    kaggle_df,
    gr_books_df,
    on='isbn13',
    how='outer',
    suffixes=('_kaggle', '_gr')
)

# Fill preferred fields
combined_df['title'] = combined_df['title_kaggle'].where(combined_df['title_kaggle'] != "", combined_df['title_gr'])
combined_df['authors'] = combined_df['authors_kaggle'].where(combined_df['authors_kaggle'] != "", combined_df['authors_gr'])
combined_df['description'] = combined_df['description_kaggle'].where(combined_df['description_kaggle'] != "", combined_df['description_gr'])
combined_df['average_rating'] = combined_df['average_rating_kaggle'].where(combined_df['average_rating_kaggle'] != "", combined_df['average_rating_gr'])
combined_df['ratings_count'] = combined_df['ratings_count_kaggle'].where(combined_df['ratings_count_kaggle'] != "", combined_df['ratings_count_gr'])
combined_df['genres'] = combined_df['categories'].where(combined_df['categories'] != "", combined_df['genres'])

# Final clean-up
final_df = combined_df[[
    'isbn13', 'title', 'authors', 'description', 'average_rating', 'ratings_count', 'genres'
]]
final_df.to_csv(OUTPUT_PATH, index=False)
print(f"✅ Merged dataset saved to: {OUTPUT_PATH}")

