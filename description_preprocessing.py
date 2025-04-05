import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("descriptions_train.csv")

print(df.columns)

descriptions = df['explanation'].dropna().tolist()

turkish_stopwords = set([
    "ve", "bir", "bu", "da", "de", "ile", "ama", "ya", "için", "çok", "gibi",
    "olarak", "diğer", "ise", "en", "ki", "mi", "mı", "mu", "mü",
    "şu", "o", "ben", "sen", "biz", "siz", "onlar", "ne", "nasıl"
])

def remove_punctuation(text):
    return re.sub(r"[^\w\sçğıöşü]", "", text)

def clean_numbers(text):
    return re.sub(r"\d+", "", text)

def remove_stopwords(text):
    words = text.split()
    return " ".join([word for word in words if word not in turkish_stopwords])

def preprocess(text):
    if isinstance(text, str):
        text = text.lower()
        text = remove_punctuation(text)
        text = clean_numbers(text)
        text = remove_stopwords(text)
        return text
    return ""

df["clean_explanation"] = df["explanation"].apply(preprocess)

print(df[["explanation", "clean_explanation"]].head())

vectorizer = TfidfVectorizer()

tfidf_matrix = vectorizer.fit_transform(df["clean_explanation"])

print("TF-IDF matris boyutu:", tfidf_matrix.shape)


cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

print("Benzerlik matrisi boyutu:", cosine_sim.shape)

def get_similar_books(book_index, cosine_sim_matrix, top_n=5):
    similarity_scores = list(enumerate(cosine_sim_matrix[book_index]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    # Kendisini çıkar (ilk sırada olur)
    similarity_scores = similarity_scores[1:top_n+1]
    return similarity_scores

similar_books = get_similar_books(0, cosine_sim)
for idx, score in similar_books:
    print(f"Kitap {idx}: {df.iloc[idx]['explanation'][:80]}... ({score:.2f})")

def show_similar_books(book_index, df, cosine_sim_matrix, top_n=5):
    similar_books = get_similar_books(book_index, cosine_sim_matrix, top_n)
    print(f"\n Seçilen Kitap [{book_index}]:\n{df.iloc[book_index]['explanation']}\n")
    print(f"En Benzer {top_n} Kitaplar:\n")
    
    for idx, score in similar_books:
        print(f"Kitap [{idx}] (Benzerlik: {score:.2f}):")
        print(df.iloc[idx]['explanation'])
        print("-" * 80)

show_similar_books(0, df, cosine_sim, top_n=5)
