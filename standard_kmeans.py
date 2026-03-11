import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.cluster import KMeans
from scipy.stats import mode

def run_standard_analysis():
    print("Downloading MNIST dataset...")
    # fetch_openml downloads the full 784-dimensional dataset (28x28 images)
    mnist = fetch_openml('mnist_784', version=1, parser='auto')
    X = mnist.data.astype('float32') / 255.0  # Normalize pixel values
    y = mnist.target.astype('int64')

    # Run K-Means with K = 10
    print("Running K-Means clustering...")
    kmeans = KMeans(n_clusters=10, random_state=42, n_init=10)
    cluster_assignments = kmeans.fit_predict(X)

    # Estimate performance and count misclassifications
    print("\n--- Clustering Results ---")
    for digit in range(10):
        # Find all instances of the current true digit
        idx = (y == digit)
        assigned_clusters_for_digit = cluster_assignments[idx]
        
        # Find the most frequent cluster assignment for this digit
        majority_cluster = mode(assigned_clusters_for_digit, keepdims=True).mode[0]
        
        # Count how many were NOT assigned to this majority cluster
        errors = np.sum(assigned_clusters_for_digit != majority_cluster)
        total = len(assigned_clusters_for_digit)
        
        print(f"Digit {digit}: Majority Cluster -> {majority_cluster} | "
              f"Misclassifications -> {errors} / {total} "
              f"({(errors/total)*100:.2f}% error)")

if __name__ == "__main__":
    run_standard_analysis()