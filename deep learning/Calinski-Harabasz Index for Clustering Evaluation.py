"""

Calinski-Harabasz Index for Clustering Evaluation
Medium
Machine Learning

Implement a function to compute the Calinski-Harabasz Index (also known as the Variance Ratio Criterion) for evaluating clustering quality. This metric measures the ratio of between-cluster dispersion to within-cluster dispersion, where higher values indicate better-defined clusters.

Given:

A 2D numpy array X of shape (n_samples, n_features) containing the data points
A 1D numpy array labels of shape (n_samples,) containing cluster assignments for each point
Return the Calinski-Harabasz score as a float. If there is only one cluster or if the number of clusters equals the number of samples, return 0.0.

Example:
Input:
X = np.array([[0, 0], [1, 0], [0, 1], [10, 10], [11, 10], [10, 11]])
labels = np.array([0, 0, 0, 1, 1, 1])
Output:
296.9985
Reasoning:
The data has 6 points split into 2 clusters. Cluster 0 contains points near origin with centroid (0.333, 0.333). Cluster 1 contains points near (10, 10) with centroid (10.333, 10.333). The global centroid is (5.333, 5.333). The within-cluster dispersion W is small (clusters are tight), while the between-cluster dispersion B is large (clusters are far apart). This results in a high CH score indicating good cluster separation.

Learn About topic
Calinski-Harabasz Index
The Calinski-Harabasz Index (CH), also known as the Variance Ratio Criterion, is an internal clustering evaluation metric that measures how well-separated and compact the clusters are.

Formula
The CH index is defined as:

C
H
=
B
/
(
k
−
1
)
W
/
(
n
−
k
)
CH= 
W/(n−k)
B/(k−1)
​
 

Where:

n
n is the total number of samples
k
k is the number of clusters
B
B is the between-cluster dispersion (inter-cluster variance)
W
W is the within-cluster dispersion (intra-cluster variance)
Computing Dispersions
Within-cluster dispersion measures how compact each cluster is:

W
=
∑
i
=
1
k
∑
x
∈
C
i
∥
x
−
μ
i
∥
2
W=∑ 
i=1
k
​
 ∑ 
x∈C 
i
​
 
​
 ∥x−μ 
i
​
 ∥ 
2
 

where 
μ
i
μ 
i
​
  is the centroid of cluster 
C
i
C 
i
​
 .

Between-cluster dispersion measures how far apart the cluster centers are from the global center:

B
=
∑
i
=
1
k
n
i
∥
μ
i
−
μ
∥
2
B=∑ 
i=1
k
​
 n 
i
​
 ∥μ 
i
​
 −μ∥ 
2
 

where 
n
i
n 
i
​
  is the number of points in cluster 
i
i, 
μ
i
μ 
i
​
  is the cluster centroid, and 
μ
μ is the global centroid of all data points.

Interpretation
Higher values indicate better clustering: clusters are dense (low 
W
W) and well-separated (high 
B
B)
Lower values suggest overlapping or dispersed clusters
The metric is not bounded above, so it's primarily used for comparing different clusterings of the same data
Edge Cases
If 
k
=
1
k=1 (single cluster), the index is undefined (we return 0)
If 
k
=
n
k=n (each point is its own cluster), 
W
=
0
W=0 which causes division issues (we return 0)
Use Cases
Determining optimal number of clusters: Plot CH score vs. 
k
k and look for the elbow or maximum
Comparing clustering algorithms: Higher CH indicates better cluster quality
Validating clustering results: Without ground truth labels

"""
import numpy as np

def calinski_harabasz_score(X: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute the Calinski-Harabasz Index for clustering evaluation.
    
    Args:
        X: numpy array of shape (n_samples, n_features) containing data points
        labels: numpy array of shape (n_samples,) containing cluster assignments
    
    Returns:
        float: Calinski-Harabasz score (higher is better)
    """
    n_samples = X.shape[0]
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    
    # Edge cases: single cluster or each point is its own cluster
    if n_clusters == 1 or n_clusters == n_samples:
        return 0.0
    
    # Compute global centroid (mean of all points)
    global_centroid = np.mean(X, axis=0)
    
    # Initialize dispersions
    between_cluster_dispersion = 0.0
    within_cluster_dispersion = 0.0
    
    # For each cluster, compute its contribution to B and W
    for cluster_id in unique_labels:
        # Get all points in this cluster
        cluster_mask = labels == cluster_id
        cluster_points = X[cluster_mask]
        n_points_in_cluster = cluster_points.shape[0]
        
        # Compute cluster centroid
        cluster_centroid = np.mean(cluster_points, axis=0)
        
        # Between-cluster dispersion: sum of n_i * ||μ_i - μ||^2
        between_cluster_dispersion += n_points_in_cluster * np.sum((cluster_centroid - global_centroid) ** 2)
        
        # Within-cluster dispersion: sum of ||x - μ_i||^2 for all x in cluster i
        within_cluster_dispersion += np.sum((cluster_points - cluster_centroid) ** 2)
    
    # Compute Calinski-Harabasz score
    # CH = [B / (k - 1)] / [W / (n - k)]
    ch_score = (between_cluster_dispersion / (n_clusters - 1)) / (within_cluster_dispersion / (n_samples - n_clusters))
    
    return ch_score
