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
"""
Understanding Eigenvalues in PCA
Principal Component Analysis (PCA) utilizes the concept of eigenvalues and eigenvectors to identify the principal components of a dataset. Here's how eigenvalues fit into the PCA process:

Eigenvalues and Eigenvectors: The Foundation of PCA
For a given square matrix ( A ), representing the covariance matrix in PCA, eigenvalues ( \lambda ) and their corresponding eigenvectors ( v ) satisfy:

A
v
=
λ
v
Av=λv
Calculating Eigenvalues
The eigenvalues of matrix ( A ) are found by solving the characteristic equation:

det
⁡
(
A
−
λ
I
)
=
0
det(A−λI)=0
where ( I ) is the identity matrix of the same dimension as ( A ). This equation highlights the relationship between a matrix, its eigenvalues, and eigenvectors.

Role in PCA
In PCA, the covariance matrix's eigenvalues represent the variance explained by its eigenvectors. Thus, selecting the eigenvectors associated with the largest eigenvalues is akin to choosing the principal components that retain the most data variance.

Eigenvalues and Dimensionality Reduction
The magnitude of an eigenvalue correlates with the importance of its corresponding eigenvector (principal component) in representing the dataset's variability. By selecting a subset of eigenvectors corresponding to the largest eigenvalues, PCA achieves dimensionality reduction while preserving as much of the dataset's variability as possible.

Practical Application
Standardize the Dataset: Ensure that each feature has a mean of 0 and a standard deviation of 1.
Compute the Covariance Matrix: Reflects how features vary together.
Find Eigenvalues and Eigenvectors: Solve the characteristic equation for the covariance matrix.
Select Principal Components: Choose eigenvectors (components) with the highest eigenvalues for dimensionality reduction.
Through this process, PCA transforms the original features into a new set of uncorrelated features (principal components), ordered by the amount of original variance they explain.

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
