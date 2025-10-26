# src/wine_feature_importance.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel


# Load Wine data
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data'
df_wine = pd.read_csv(url, header=None)

df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash',
                   'Alcalinity of ash', 'Magnesium', 'Total phenols',
                   'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
                   'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline']

print("Wine dataset shape:", df_wine.shape)
print(df_wine.head())

# Prepare features and labels
X = df_wine.iloc[:, 1:].values
y = df_wine.iloc[:, 0].values

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0, stratify=y
)

# Normalize then standardize (as in your original)
mms = MinMaxScaler()
X_train_norm = mms.fit_transform(X_train)
X_test_norm = mms.transform(X_test)

stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train_norm)
X_test_std = stdsc.transform(X_test_norm)

# Train Random Forest
feature_labels = df_wine.columns[1:]
forest = RandomForestClassifier(n_estimators=500, random_state=1)
forest.fit(X_train_std, y_train)

# Feature importance
importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]

print("\nFeature Ranking:")
for f in range(X_train_std.shape[1]):
    print(f"{f+1:2d}) {feature_labels[indices[f]]:30} {importances[indices[f]]:.6f}")

# Plot importance
plt.figure(figsize=(10, 6))
plt.title('Feature Importance - Wine Dataset')
plt.bar(range(X_train_std.shape[1]), importances[indices], align='center')
plt.xticks(range(X_train_std.shape[1]), [feature_labels[i] for i in indices], rotation=90)
plt.xlim([-1, X_train_std.shape[1]])
plt.tight_layout()
plt.show()

# Feature selection
sfm = SelectFromModel(forest, threshold=0.1, prefit=True)
X_selected = sfm.transform(X_train_std)  # ← FIXED: use standardized data!

print(f"\nNumber of features selected (threshold=0.1): {X_selected.shape[1]}")
selected_features = feature_labels[indices][:X_selected.shape[1]]
for i, feat in enumerate(selected_features):
    print(f"{i+1:2d}) {feat:30} {importances[indices[i]]:.6f}")