import numpy as np
import matplotlib.pyplot as plt

class KMeansClustur:
    def __init__(self, k):
        self.k = k
        self.centroids = None
        self.assigned_labels = None

    def euclidean(self, x, centroids):
        """
        Parameters:
            x is d by 1 array
            centroids is d by k array
        Returns :
            (n,) array: the pairwise Euclidean distance of x with 
            respect to individual samples in  centroids
        """
        return np.sqrt(np.sum((centroids-x)**2, axis=0, keepdims=True))**2


    def k_means(self, X, k, iter):
        D, N = X.shape[0], X.shape[1]
        indices = np.random.choice(X.shape[1], k, replace=False)
        centroids = np.take(X, indices, axis=1)
        #print(centroids)
        y = np.full((1, X.shape[1]), np.random.choice(X.shape[1]))
        for it in range(iter):
            y_old = y.copy()
            for i in range(N):
                y[0][i] = np.argmin(self.euclidean(X[:, i:i+1], centroids))
                #print(y)
            for j in range(k):
                mean = np.sum(np.where(y==j, X, 0), axis=1, keepdims=True)/ \
                        np.sum(np.where(y==j, 1, 0), axis=1, keepdims=True)
                centroids[:, j:j+1] = mean
                #print(centroids)
            if np.array_equiv(y, y_old):
                break
        return centroids, y

    def run_k_means(self, X, iter=100000):
        self.centroids, self.assigned_labels = \
            self.k_means(X, self.k, iter)

    #Assigns a new data point to one of the clusters
    def assign_label(self, x):
        return np.argmin(self.euclidean(x, self.centroids))

#####################test######################
# from sklearn.datasets import make_blobs

# centers = [(10, 10), (10, 100), (70, 110), (100, 10), (120, 70)]
# cluster_std = [10, 15, 5, 15, 7]
# # x1, x2, x3, x4, x5

# X, y = make_blobs(n_samples=300, cluster_std=cluster_std, 
#                                 centers=centers, n_features=2, random_state=1)



# k=5
# km = KMeansClustur(k)
# X = X.T
# km.run_k_means(X)

# x = np.array([[20], [25]])

# print(km.assign_label(x))

# def plot_k_means(X, km):
#     c, y = km.centroids, km.assigned_labels
#     plt.figure(facecolor="white")
#     ax = plt.subplot()
#     ax.scatter(X[0:1, :], X[1:2, :], 3, y[0:1, :], cmap='plasma')
#     n = [i for i in range(k)]
#     for i, txt in enumerate(n):
#         ax.annotate(txt, (c[0:1, i:i+1], c[1:2, i:i+1]))
#     plt.show()

# plot_k_means(X, km)