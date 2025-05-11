# app/books_data/load_data.py

import pandas as pd

def load_interactions_matrix():
    # Dummy interactions (user_id → book title → rating)
    data = {
        1: {"Book A": 5, "Book B": 3, "Book C": None},
        2: {"Book A": 4, "Book C": 5},
        3: {"Book A": None, "Book B": 4, "Book C": 3}
    }
    return pd.DataFrame(data).T  # users as rows, books as columns
