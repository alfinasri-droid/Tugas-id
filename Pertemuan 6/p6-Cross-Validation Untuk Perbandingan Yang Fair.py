from sklearn.model_selection import cross_val_score, KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import time
import pandas as pd

# Asumsikan X, y sudah terdefinisi (misal dari load_breast_cancer)
# Jika belum, tambahkan:
# from sklearn.datasets import load_breast_cancer
# data = load_breast_cancer()
# X, y = data.data, data.target

# Definisikan model stacking (jika belum ada)
base_models = [
    ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
    ('svm', SVC(kernel='rbf', probability=True, random_state=42)),
    ('knn', KNeighborsClassifier(n_neighbors=5)),
    ('dt', DecisionTreeClassifier(max_depth=5, random_state=42))
]
meta_learner = LogisticRegression(max_iter=1000)
stacking = StackingClassifier(
    estimators=base_models,
    final_estimator=meta_learner,
    cv=5,
    stack_method='predict_proba'
)

# Daftar model yang akan dibandingkan
models = {
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'AdaBoost': AdaBoostClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'Stacking': stacking
}

# Cross-validation dengan 5-fold (shuffle)
cv = KFold(n_splits=5, shuffle=True, random_state=42)

results = []
for name, model in models.items():
    start_time = time.time()
   # scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    elapsed_time = time.time() - start_time

    results.append({
        'Model': name,
       # 'Mean Accuracy': scores.mean(),
      #  'Std': scores.std(),
        'Training Time (s)': elapsed_time
    })

# Tampilkan hasil dalam DataFrame
results_df = pd.DataFrame(results).round(4)
print(results_df.to_string(index=False))

# Kesimpulan model terbaik
best_model = results_df.loc[results_df['Mean Accuracy'].idxmax(), 'Model']
print(f"\nModel terbaik berdasarkan CV: {best_model}")