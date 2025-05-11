# app/api/routes.py
from fastapi import APIRouter
from pydantic import BaseModel
from typing import List
from app.recommender.engine import get_recommendations

recommend_router = APIRouter()

class RecommendRequest(BaseModel):
    title: str
    genres: dict  # e.g., {"Horror": 4, "Science Fiction": 2}

class RecommendResponse(BaseModel):
    recommended_titles: List[str]

@recommend_router.post("/", response_model=RecommendResponse)
def recommend(request: RecommendRequest):
    recs = get_recommendations(request.title, request.genres)
    return {"recommended_titles": recs}
