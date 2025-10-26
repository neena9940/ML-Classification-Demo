# src/iris_classifier.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap

# Load and prepare data
iris = datasets.load_iris()
X = iris.data[:, [2, 3]]  # petal length, petal width
y = iris.target

# Imputation (though Iris has no missing values, kept for demo)
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1, stratify=y
)

# Standardize
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

# Combine for plotting
X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

# Plot function
def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
    markers = ('o', 's', '^', 'v', '<')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))

    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=f'Class {cl}',
                    edgecolor='black')
    if test_idx is not None:
        X_test_plot = X[test_idx, :]
        plt.scatter(X_test_plot[:, 0],
                    X_test_plot[:, 1],
                    c='none', edgecolor='black',
                    alpha=1.0, linewidths=1.0,
                    marker='o', s=100, label='Test set')

# Train and evaluate models
models = {}

# Perceptron
ppn = Perceptron(eta0=0.1, random_state=1, max_iter=1000)
ppn.fit(X_train_std, y_train)
models['Perceptron'] = ppn

# Logistic Regression
lr = LogisticRegression(C=100, solver='lbfgs', multi_class='ovr', max_iter=1000)
lr.fit(X_train_std, y_train)
models['Logistic Regression'] = lr

# SVM Linear
svm_linear = SVC(kernel='linear', C=1.0, random_state=1)
svm_linear.fit(X_train_std, y_train)
models['SVM (Linear)'] = svm_linear

# SVM RBF
svm_rbf = SVC(kernel='rbf', C=1.0, gamma=0.2, random_state=1)
svm_rbf.fit(X_train_std, y_train)
models['SVM (RBF)'] = svm_rbf

# Decision Tree
tree_model = DecisionTreeClassifier(criterion='gini', max_depth=4, random_state=1)
tree_model.fit(X_train_std, y_train)
models['Decision Tree'] = tree_model

# Random Forest
forest = RandomForestClassifier(n_estimators=25, random_state=1, n_jobs=-1)
forest.fit(X_train_std, y_train)
models['Random Forest'] = forest

# KNN
knn = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')
knn.fit(X_train_std, y_train)
models['KNN'] = knn

# Print accuracies
print("Model Accuracies on Iris Test Set:")
for name, model in models.items():
    acc = accuracy_score(y_test, model.predict(X_test_std))
    print(f"{name:20}: {acc:.3f}")

# Plot one example (e.g., Random Forest)
plt.figure(figsize=(8, 6))
plot_decision_regions(X_combined_std, y_combined, classifier=forest,
                      test_idx=range(105, 150))
plt.xlabel('Petal length [standardized]')
plt.ylabel('Petal width [standardized]')
plt.legend(loc='upper left')
plt.title('Random Forest - Iris Dataset')
plt.tight_layout()
plt.show()

# Plot decision tree
plt.figure(figsize=(12, 8))
feature_names = ['Sepal length', 'Sepal width', 'Petal length', 'Petal width']
plot_tree(tree_model, feature_names=feature_names, class_names=iris.target_names,
          filled=True, rounded=True)
plt.title("Decision Tree Visualization")
plt.show()