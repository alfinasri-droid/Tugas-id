import numpy as np
from sklearn.datasets import load_iris

# ──────────────────────────────────────────────
# SOAL 3 — Cosine Similarity untuk Teks
# Modifikasi gaya: Implementasi Jarak Numerik
#                  dengan NumPy
# ──────────────────────────────────────────────

# Tiga dokumen
# D1: "data science is fun"
# D2: "data mining is fun"
# D3: "machine learning is cool"

# Vocabulary (sorted): cool, data, fun, is, learning, machine, mining, science
#                         0     1    2   3      4        5        6       7

D1 = np.array([0, 1, 1, 1, 0, 0, 0, 1])  # data science is fun
D2 = np.array([0, 1, 1, 1, 0, 0, 1, 0])  # data mining is fun
D3 = np.array([1, 0, 0, 1, 1, 1, 0, 0])  # machine learning is cool

# Fungsi cosine similarity manual (NumPy)
def cosine_similarity_np(a, b):
    dot_product  = np.dot(a, b)
    norm_a       = np.linalg.norm(a)
    norm_b       = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b)

# Hitung cosine similarity antara D1-D2 dan D1-D3
cos_d1_d2 = cosine_similarity_np(D1, D2)
cos_d1_d3 = cosine_similarity_np(D1, D3)

print("=" * 45)
print("  COSINE SIMILARITY — SOAL 3")
print("=" * 45)
print(f"  cos(D1, D2) : {cos_d1_d2:.4f}")   # 0.7500
print(f"  cos(D1, D3) : {cos_d1_d3:.4f}")   # 0.2500
print("=" * 45)

# ──────────────────────────────────────────────
# EKSTENSI: iris.data + Looping
# Struktur yang sama, data numerik nyata
# ──────────────────────────────────────────────

iris = load_iris()
X    = iris.data        # shape: (150, 4)

print("\nCosine Similarity iris — row[0] vs setiap baris:")
print("-" * 45)

for i in range(len(X)):
    sim = cosine_similarity_np(X[0], X[i])
    print(f"  row[0] vs row[{i:3d}] : {sim:.4f}")

# Similarity matrix lengkap (semua pasang)
n          = len(X)
sim_matrix = np.zeros((n, n))

for i in range(n):
    for j in range(n):
        sim_matrix[i, j] = cosine_similarity_np(X[i], X[j])

print("\nSimilarity Matrix (5x5 preview):")
print(sim_matrix[:5, :5].round(4))