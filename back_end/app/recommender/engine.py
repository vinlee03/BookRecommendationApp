# app/recommender/engine.py

from app.recommender.content import content_based_filtering
from app.recommender.collaborative import collaborative_filtering
from app.recommender.regression import probability_score
from app.recommender.reduce import reduce_dimensions

def get_recommendations(title: str, genres: dict) -> list:
    # 1. Content-based recs
    content_recs = content_based_filtering(title)

    # 2. Collaborative filtering (on title or user profile)
    collab_recs = collaborative_filtering(title)

    # 3. Combine + score via regression
    combined = list(set(content_recs + collab_recs))
    scored = probability_score(title, combined)

    # 4. Optionally: reduce for clustering or grouping
    final_recs = reduce_dimensions(scored)

    # Return top 5
    return [item["title"] for item in final_recs[:5]]
