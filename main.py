from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from cosine_sim import get_similar_books
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Book Recommendation API",
    description="Kitap açıklamalarına göre içerik tabanlı öneri sistemi.",
    version="1.0.0"
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Tüm kaynaklardan gelen istekleri kabul eder
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class RecommendationRequest(BaseModel):
    visited_ids: List[int]

class RecommendationResponse(BaseModel):
    recommended_ids: List[int]

@app.get("/")
def root():
    return {"message": "Recommendation service is running"}

@app.post("/recommend", response_model=RecommendationResponse)
def recommend_books(request: RecommendationRequest):
    print_requests(request.visited_ids)
    similar_ids = get_similar_books(request.visited_ids)
    print_requests(similar_ids)
    return RecommendationResponse(recommended_ids=similar_ids)

def print_requests(array: List[int]):
    for i in array:
        print(i)