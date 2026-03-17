from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import numpy as np
# Load dataset
data = load_iris()
X = data.data
y = data.target
colors = np.array(['red','lime','black'])
plt.figure(figsize=(12,4))
# Real clusters
plt.subplot(1,3,1)
plt.scatter(X[:,2], X[:,3], c=colors[y])
plt.title("Real")
# K-Means clustering
kmeans = KMeans(n_clusters=3)
kmeans_labels = kmeans.fit_predict(X)
plt.subplot(1,3,2)
plt.scatter(X[:,2], X[:,3], c=colors[kmeans_labels])
plt.title("K-Means")
# EM (Gaussian Mixture Model)
gmm = GaussianMixture(n_components=3)
gmm_labels = gmm.fit_predict(X)
plt.subplot(1,3,3)
plt.scatter(X[:,2], X[:,3], c=colors[gmm_labels])
plt.title("EM (GMM)")
plt.show()