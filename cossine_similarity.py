from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

vectorizer = TfidfVectorizer()

tfidf_matrix = vectorizer.fit_transform(df["clean_explanation"])

print("TF-IDF matris boyutu:", tfidf_matrix.shape)


cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

print("Benzerlik matrisi boyutu:", cosine_sim.shape)

def get_similar_books(book_index, cosine_sim_matrix, top_n=5):
    similarity_scores = list(enumerate(cosine_sim_matrix[book_index]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    similarity_scores = similarity_scores[1:top_n+1]
    return similarity_scores


similar_books = get_similar_books(0, cosine_sim)
for idx, score in similar_books:
    print(f"Kitap {idx}: {df.iloc[idx]['explanation'][:80]}... ({score:.2f})")
