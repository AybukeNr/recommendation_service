from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from cosine_sim import get_similar_books

app = FastAPI(
    title="Book Recommendation API",
    description="Kitap açıklamalarına göre içerik tabanlı öneri sistemi.",
    version="1.0.0"
)


class RecommendationRequest(BaseModel):
    visited_ids: List[int]

class RecommendationResponse(BaseModel):
    recommended_ids: List[int]


@app.post("/recommend", response_model=RecommendationResponse)
def recommend_books(request: RecommendationRequest):
    similar_ids = get_similar_books(request.visited_ids)
    return RecommendationResponse(recommended_ids=similar_ids)

