import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml

# 1. Custom SVD-based Dimensionality Reduction
def custom_svd_pca(X, n_components=50):
    # Center the data
    X_mean = np.mean(X, axis=0)
    X_centered = X - X_mean
    
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    
    components = Vt[:n_components]
    X_reduced = X_centered @ components.T
    
    return X_reduced

# 2. Custom K-Means using Matrix Operations
def custom_kmeans(X, K=10, max_iters=100, tol=1e-4, random_state=42):
    np.random.seed(random_state)
    N, D = X.shape
    
    # Initialize centroids by randomly picking K data points
    random_indices = np.random.choice(N, K, replace=False)
    centroids = X[random_indices].copy()
    X_sq = np.sum(X**2, axis=1, keepdims=True)
    labels = np.zeros(N, dtype=int)
    
    for iteration in range(max_iters):
        # Calculate ||c||^2 (1 x K matrix)
        C_sq = np.sum(centroids**2, axis=1)
        
        # Calculate 2 * x * c^T (N x K matrix)
        XC = X @ centroids.T
        
        # Vectorized Euclidean distance matrix: ||x||^2 + ||c||^2 - 2(x dot c)
        distances = X_sq + C_sq - 2 * XC
        
        # Assign points to the closest centroid
        new_labels = np.argmin(distances, axis=1)
        
        # Check for convergence
        if np.array_equal(labels, new_labels):
            print(f"K-Means converged at iteration {iteration}")
            break
            
        labels = new_labels
        
        # Update centroids
        new_centroids = np.zeros_like(centroids)
        for k in range(K):
            cluster_points = X[labels == k]
            if len(cluster_points) > 0:
                new_centroids[k] = np.mean(cluster_points, axis=0)
            else:
                # Re-initializing to a random point if cluster is empty
                new_centroids[k] = X[np.random.choice(N)]
                
        # Check if centroids shifted within tolerance
        shift = np.linalg.norm(centroids - new_centroids, ord='fro')
        centroids = new_centroids
        if shift < tol:
            print(f"K-Means converged at iteration {iteration} based on tolerance.")
            break
            
    return labels

def run_custom_analysis():
    print("Downloading and preparing MNIST for custom analysis...")
    mnist = fetch_openml('mnist_784', version=1, parser='auto')
    X = mnist.data.to_numpy().astype('float32') / 255.0
    y = mnist.target.to_numpy().astype('int64')

    print("Reducing dimensions via SVD...")
    X_reduced = custom_svd_pca(X, n_components=50)

    print("Running custom vectorized K-Means...")
    cluster_assignments = custom_kmeans(X_reduced, K=10)

    print("\n--- Custom K-Means Results ---")
    for digit in range(10):
        idx = (y == digit)
        assigned_clusters_for_digit = cluster_assignments[idx]
        
        if len(assigned_clusters_for_digit) > 0:
            majority_cluster = np.bincount(assigned_clusters_for_digit).argmax()
            errors = np.sum(assigned_clusters_for_digit != majority_cluster)
            total = len(assigned_clusters_for_digit)
            
            print(f"Digit {digit}: Majority Cluster -> {majority_cluster} | "
                  f"Misclassifications -> {errors} / {total} "
                  f"({(errors/total)*100:.2f}% error)")

if __name__ == "__main__":
    run_custom_analysis()