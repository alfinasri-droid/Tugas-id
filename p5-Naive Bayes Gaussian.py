# IMPLEMENTASI NAIVE BAYES DENGAN SCIKIT-LEARN
# KODE: GAUSSIAN NAIVE BAYES
# LISTING 1 & 2: naive_bayes_gaussian.py

import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.datasets import load_iris
#import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Gaussian Naive Bayes (untuk data numerik)
gnb = GaussianNB()
gnb.fit(X_train, y_train)

# Prediksi
y_pred = gnb.predict(X_test)

# Evaluasi
print(f"Akurasi: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
#sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
#            xticklabels=iris.target_names,
#            yticklabels=iris.target_names)
plt.title('Confusion Matrix - Naive Bayes')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# Lihat probabilitas posterior
probabilities = gnb.predict_proba(X_test[:5])
print("\nProbabilitas untuk 5 data pertama:")
print(pd.DataFrame(probabilities, columns=iris.target_names))