from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Kumpulan dokumen
documents = [
    "data mining machine learning",
    "machine learning data mining",
    "artificial intelligence",
    "deep learning neural network"
]

# Konversi ke vektor TF-IDF
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)

# Hitung cosine similarity antara semua dokumen
similarity_matrix = cosine_similarity(tfidf_matrix)

print("Cosine Similarity Matrix:")
print(similarity_matrix.round(3))
print("\nFitur (kata-kata):")
print(vectorizer.get_feature_names_out())

# Cari dokumen paling mirip dengan dokumen pertama (indeks 0), abaikan dirinya sendiri
sim_to_first = similarity_matrix[0].copy()
sim_to_first[0] = -1  # abaikan similarity dengan diri sendiri
most_similar_idx = np.argmax(sim_to_first)
print(f"\nDokumen 0 paling mirip dengan dokumen {most_similar_idx}")