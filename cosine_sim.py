from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def compute_similar_books(input_texts, all_texts_df, top_n=5):
    all_texts = list(all_texts_df["clean_explanation"])
    all_ids = list(all_texts_df["id"])

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(all_texts + input_texts)

    cosine_sim = cosine_similarity(tfidf_matrix[-len(input_texts):], tfidf_matrix[:-len(input_texts)])

    similar_ids = set()
    for row in cosine_sim:
        top_matches = sorted(enumerate(row), key=lambda x: x[1], reverse=True)[:top_n]
        for idx, _ in top_matches:
            similar_ids.add(all_ids[idx])
    return list(similar_ids)
