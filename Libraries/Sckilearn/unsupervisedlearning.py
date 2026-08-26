# 1. Load the Iris dataset (they didn’t even show this!)
from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data  # Features (sepal length, sepal width, petal length, petal width)

# 2. Apply K-Means
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X)

# 3. Plot the clusters (only using the first two features)
import matplotlib.pyplot as plt  # Correct import (they wrote 'matplotlib as plt'—wrong!)
plt.scatter(X[:, 0], X[:, 1], c=clusters, cmap='viridis')
plt.xlabel(iris.feature_names[0])  # e.g., "sepal length (cm)"
plt.ylabel(iris.feature_names[1])  # e.g., "sepal width (cm)"
plt.title("K-Means Clustering on Iris Dataset (K=3)")
plt.show()