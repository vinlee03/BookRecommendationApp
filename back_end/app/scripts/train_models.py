import pandas as pd
import pickle
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

# Paths
DATA_PATH = Path(__file__).resolve().parents[1] / "books_data" / "merged_books.csv"
ASSET_DIR = Path(__file__).resolve().parents[1] / "assets"
ASSET_DIR.mkdir(exist_ok=True)

# Load dataset
df = pd.read_csv(DATA_PATH)
df['description'] = df['description'].fillna("")

# TF-IDF
tfidf = TfidfVectorizer(stop_words='english', max_features=10000)
tfidf_matrix = tfidf.fit_transform(df['description'])

# Save TF-IDF vectorizer
with open(ASSET_DIR / "tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(tfidf, f)

# Save matrix if needed
from scipy.sparse import save_npz
save_npz(ASSET_DIR / "tfidf_matrix.npz", tfidf_matrix)

# Dimensionality reduction (optional)
svd = TruncatedSVD(n_components=100)
reduced_matrix = svd.fit_transform(tfidf_matrix)

with open(ASSET_DIR / "svd_model.pkl", "wb") as f:
    pickle.dump(svd, f)

# Save reduced matrix
import numpy as np
np.save(ASSET_DIR / "reduced_matrix.npy", reduced_matrix)

print("✅ Models saved to:", ASSET_DIR)
