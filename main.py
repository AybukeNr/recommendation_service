from fastapi import FastAPI
from pydantic import BaseModel
from books_db import get_descriptions_by_ids
from cosine_sim import compute_similar_books
import pandas as pd

app = FastAPI()

all_books_df = pd.read_csv("descriptions_clean.csv")  

class BookIDRequest(BaseModel):
    visited_ids: list[int]

@app.post("/recommend")
def recommend_books(req: BookIDRequest):
    descriptions = get_descriptions_by_ids(req.visited_ids)
    input_clean_texts = [desc for (_, desc) in descriptions]  
    similar_ids = compute_similar_books(input_clean_texts, all_books_df)
    return {"recommended_ids": similar_ids}
