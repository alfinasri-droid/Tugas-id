import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# ==================== LOAD DATASET ====================
wine = load_wine()
X = wine.data
y = wine.target

# ==================== NORMALISASI (PENTING UNTUK K-NN) ====================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ==================== SPLIT DATA ====================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)

# ==================== K-NN DENGAN k=5 ====================
knn = KNeighborsClassifier(n_neighbors=5, weights='uniform')
knn.fit(X_train, y_train)

# Prediksi
y_pred = knn.predict(X_test)
print(f"Akurasi (k=5): {accuracy_score(y_test, y_pred):.4f}")

# ==================== TUNING PARAMETER k DENGAN CROSS-VALIDATION ====================
param_grid = {
    'n_neighbors': [3, 5, 7, 9, 11, 13, 15],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'minkowski']
}

grid_search = GridSearchCV(
    KNeighborsClassifier(),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

# ==================== HASIL TUNING ====================
print(f"\nParameter terbaik: {grid_search.best_params_}")
print(f"Akurasi CV terbaik: {grid_search.best_score_:.4f}")

# ==================== EVALUASI MODEL TERBAIK ====================
best_knn = grid_search.best_estimator_
y_pred_best = best_knn.predict(X_test)
print(f"Akurasi test set: {accuracy_score(y_test, y_pred_best):.4f}")