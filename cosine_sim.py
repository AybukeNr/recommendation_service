import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from books_db import get_recommended_books, get_visited_books_by_ids


turkish_stopwords = set([
    "ve", "bir", "bu", "da", "de", "ile", "ama", "ya", "için", "çok", "gibi",
    "olarak", "diğer", "ise", "en", "ki", "mi", "mı", "mu", "mü",
    "şu", "o", "ben", "sen", "biz", "siz", "onlar", "ne", "nasıl"
])

def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^\w\sçğıöşü]", "", text)
    text = re.sub(r"\d+", "", text)
    words = text.split()
    return " ".join([word for word in words if word not in turkish_stopwords])

def build_similarity_matrix():
    books = get_recommended_books()
    explanations = [preprocess(book['description']) for book in books]
    ids = [book['id'] for book in books]

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(explanations)
    similarity_matrix = cosine_similarity(tfidf_matrix)

    return ids, similarity_matrix

def get_similar_books(target_ids, top_n=5):
    all_books = get_recommended_books()
    ids, similarity_matrix = build_similarity_matrix()

    id_to_index = {bid: idx for idx, bid in enumerate(ids)}
    index_to_id = {idx: bid for idx, bid in enumerate(ids)}

    similarity_scores = np.zeros(len(ids))

    for tid in target_ids:
        if tid in id_to_index:
            idx = id_to_index[tid]
            similarity_scores += similarity_matrix[idx]

    # Ziyaret edilmiş kitaplar tekrar önerilmesin diye idleri 0 yapıyoruz
    for tid in target_ids:
        if tid in id_to_index:
            similarity_scores[id_to_index[tid]] = 0

    top_indices = similarity_scores.argsort()[::-1][:top_n]
    recommended_ids = [index_to_id[i] for i in top_indices]
    return recommended_ids
