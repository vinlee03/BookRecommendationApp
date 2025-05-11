import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline

def reduce_tfidf_matrix(tfidf_matrix, n_components=100):
    """Apply Truncated SVD to reduce TF-IDF matrix dimensions."""
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    normalizer = Normalizer(copy=False)
    lsa = make_pipeline(svd, normalizer)

    reduced_matrix = lsa.fit_transform(tfidf_matrix)
    return reduced_matrix, svd
