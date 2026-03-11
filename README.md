# MNIST Clustering & Dimensionality Reduction

This repository compares a **Standard Scikit-Learn** approach against a **Custom implementation** of K-Means and PCA (via SVD) for clustering the MNIST handwritten digit dataset.

## Features
- **Dimensionality Reduction:** Custom SVD implementation to reduce feature space from 784 to 50 components.
- **Vectorized Operations:** K-Means implemented using matrix broadcasting for optimized distance calculations.
- **Performance Evaluation:** Automated majority-label assignment to calculate error rates per digit.

## Usage
1. Install dependencies: `pip install -r requirements.txt`
2. Run the standard analysis: `python standard_kmeans.py`
3. Run the custom implementation: `python custom_kmeans_pca.py`