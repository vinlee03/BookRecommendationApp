import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from pathlib import Path

# Optional: Uncomment code below to use dimensionality reduction
# from app.recommender.reduce import reduce_tfidf_matrix

books_path = Path(__file__).resolve().parent.parent / "books_data" / "merged_books.csv"
books = pd.read_csv(books_path)
books['description'] = books['description'].fillna("")

# TF-IDF
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(books['description'])

# Optional: Uncomment code below to use SVD reduction
# reduced_matrix, svd_model = reduce_tfidf_matrix(tfidf_matrix)

# Book index
indices = pd.Series(books.index, index=books['title']).drop_duplicates()

def content_based_filtering_scores(title: str, top_n: int = 10, use_svd: bool = False) -> dict:
    """Returns top similar books by description content."""
    if title not in indices:
        return {}

    idx = indices[title]

    # Choose between full TF-IDF or reduced
    matrix = tfidf_matrix  # default
    if use_svd:
        from app.recommender.reduce import reduce_tfidf_matrix
        reduced_matrix, svd_model = reduce_tfidf_matrix(tfidf_matrix)
        matrix = reduced_matrix
        # Transform the query vector too
        query_vec = svd_model.transform(tfidf_matrix[idx])
    else:
        query_vec = tfidf_matrix[idx:idx+1]

    cosine_sim = linear_kernel(query_vec, matrix).flatten()
    sim_scores = list(enumerate(cosine_sim))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    return {books.iloc[i]['title']: score for i, score in sim_scores}
