import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# ==================== LOAD DATA ====================
data = load_breast_cancer()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# ==================== SINGLE DECISION TREE ====================
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)

# ==================== BAGGING ====================
bagging = BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)
bagging.fit(X_train, y_train)
bagging_pred = bagging.predict(X_test)

# ==================== RANDOM FOREST ====================
rf = RandomForestClassifier(
    n_estimators=100,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

# ==================== ADABOOST ====================
ada = AdaBoostClassifier(n_estimators=100, random_state=42)
ada.fit(X_train, y_train)
ada_pred = ada.predict(X_test)

# ==================== GRADIENT BOOSTING ====================
gb = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)
gb.fit(X_train, y_train)
gb_pred = gb.predict(X_test)

# ==================== STACKING ====================
# Base models
base_models = [
    ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
    ('svm', SVC(kernel='rbf', probability=True, random_state=42)),
    ('knn', KNeighborsClassifier(n_neighbors=5)),
    ('dt', DecisionTreeClassifier(max_depth=5, random_state=42))
]

# Meta-learner
meta_learner = LogisticRegression(max_iter=1000)

# Stacking Classifier
stacking = StackingClassifier(
    estimators=base_models,
    final_estimator=meta_learner,
    cv=5,                      # cross-validation untuk generate meta-features
    stack_method='predict_proba'  # untuk klasifikasi
)
stacking.fit(X_train, y_train)
stacking_pred = stacking.predict(X_test)

# ==================== PERBANDINGAN SEMUA METODE ====================
print("=" * 50)
print("PERBANDINGAN SEMUA METODE")
print("=" * 50)
print(f"Single Decision Tree: {accuracy_score(y_test, dt_pred):.4f}")
print(f"Bagging: {accuracy_score(y_test, bagging_pred):.4f}")
print(f"Random Forest: {accuracy_score(y_test, rf_pred):.4f}")
print(f"AdaBoost: {accuracy_score(y_test, ada_pred):.4f}")
print(f"Gradient Boosting: {accuracy_score(y_test, gb_pred):.4f}")
print(f"Stacking: {accuracy_score(y_test, stacking_pred):.4f}")

# ==================== VISUALISASI PERBANDINGAN ====================
models = ['DT', 'Bagging', 'RF', 'AdaBoost', 'GB', 'Stacking']
scores = [
    accuracy_score(y_test, dt_pred),
    accuracy_score(y_test, bagging_pred),
    accuracy_score(y_test, rf_pred),
    accuracy_score(y_test, ada_pred),
    accuracy_score(y_test, gb_pred),
    accuracy_score(y_test, stacking_pred)
]

plt.figure(figsize=(10, 5))
plt.bar(models, scores, color=['blue', 'green', 'red', 'orange', 'purple', 'brown'])
plt.ylabel('Accuracy')
plt.title('Perbandingan Akurasi: Single Model vs Ensemble')
plt.ylim(0.9, 1.0)
for i, v in enumerate(scores):
    plt.text(i, v + 0.002, f'{v:.4f}', ha='center')
plt.show()