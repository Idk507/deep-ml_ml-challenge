"""
Write a Python function that performs Principal Component Analysis (PCA) from scratch. The function should take a 2D NumPy array as input, where each row represents a data sample and each column represents a feature. The function should standardize the dataset, compute the covariance matrix, find the eigenvalues and eigenvectors, and return the principal components (the eigenvectors corresponding to the largest eigenvalues). The function should also take an integer k as input, representing the number of principal components to return.

Example:
Input:
data = np.array([[1, 2], [3, 4], [5, 6]]), k = 1
Output:
[[0.7071], [0.7071]]
Reasoning:
After standardizing the data and computing the covariance matrix, the eigenvalues and eigenvectors are calculated. The largest eigenvalue's corresponding eigenvector is returned as the principal component, rounded to four decimal places.


"""
import numpy as np

def pca(data, k):
    # Standardize the data
    data_standardized = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
    
    # Compute the covariance matrix
    covariance_matrix = np.cov(data_standardized, rowvar=False)
    
    # Eigen decomposition
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
    
    # Sort the eigenvectors by decreasing eigenvalues
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues_sorted = eigenvalues[idx]
    eigenvectors_sorted = eigenvectors[:,idx]
    
    # Select the top k eigenvectors (principal components)
    principal_components = eigenvectors_sorted[:, :k]
    
    return np.round(principal_components, 4)
