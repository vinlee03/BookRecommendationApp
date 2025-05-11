from app.recommender.content import content_based_filtering_scores
from app.recommender.collaborative import collaborative_filtering_scores
import pandas as pd

def ensemble_predict(title: str, user_id: int, alpha: float = 0.5):
    # alpha is the weight: 0 = only collab, 1 = only content
    content_scores = content_based_filtering_scores(title)
    collab_scores = collaborative_filtering_scores(user_id)

    # Convert to DataFrames
    content_df = pd.DataFrame(list(content_scores.items()), columns=["title", "content_score"])
    collab_df = pd.DataFrame(list(collab_scores.items()), columns=["title", "collab_score"])

    # Merge on title
    merged = pd.merge(content_df, collab_df, on="title", how="outer").fillna(0)
    merged["ensemble_score"] = alpha * merged["content_score"] + (1 - alpha) * merged["collab_score"]

    return merged.sort_values("ensemble_score", ascending=False).head(5)[["title", "ensemble_score"]].to_dict(orient="records")
