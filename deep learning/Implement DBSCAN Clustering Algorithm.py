"""
Implement DBSCAN Clustering AlgorithmWrite a Python function dbscan that implements the Density-Based Spatial Clustering of Applications with Noise (DBSCAN) algorithm. The function should take a 2D numpy array X representing the data points of shape (n_samples, n_features), a float eps representing the maximum distance between two samples for them to be considered neighbors, and an integer min_samples representing the minimum number of points required to form a dense region (core point). The function should return a 1D numpy array of cluster labels where -1 indicates noise points (points that don't belong to any cluster). Use Euclidean distance for computing distances between points. The algorithm should identify core points, expand clusters from core points, and mark remaining points as noise.

Example:

Input:

X = np.array([[1, 1], [1.5, 1.5], [2, 1], [8, 8], [8.5, 8.5], [9, 8]])
eps = 2.0
min_samples = 2
print(dbscan(X, eps, min_samples))

Output:

[0 0 0 1 1 1]

Reasoning:

Points [1,1], [1.5,1.5], and [2,1] are close together (within eps=2.0 of each other) and each has at least 2 neighbors, so they form cluster 0. Similarly, points [8,8], [8.5,8.5], and [9,8] are close together and form cluster 1. No points are marked as noise since all belong to dense regions.

Learn About topic

DBSCAN Clustering Algorithm

Overview

DBSCAN (Density-Based Spatial Clustering of Applications with Noise) is a popular clustering algorithm that groups together points that are closely packed together, marking points in low-density regions as outliers (noise).

Key Concepts

1. Core Points A point pp is a core point if at least min_samples points (including pp itself) are within distance ϵϵ (eps) from it: ∣Nϵ(p)∣≥min_samples∣Nϵ​(p)∣≥min_samples where Nϵ(p)={q∈D:dist(p,q)≤ϵ}Nϵ​(p)={q∈D:dist(p,q)≤ϵ}

2. Directly Density-Reachable A point qq is directly density-reachable from pp if:





pp is a core point



q∈Nϵ(p)q∈Nϵ​(p)

3. Density-Reachable A point qq is density-reachable from pp if there exists a chain of points p1,p2,...,pnp1​,p2​,...,pn​ where p1=pp1​=p, pn=qpn​=q, and each pi+1pi+1​ is directly density-reachable from pipi​.

4. Noise Points Points that are not reachable from any core point are labeled as noise (-1).

Algorithm Steps





For each unvisited point pp:





Mark pp as visited



Find all neighbors within ϵϵ



If ∣neighbors∣<min_samples∣neighbors∣<min_samples: mark as noise (may change later)



Else: create new cluster and expand it



Expand cluster:





Add all neighbors to cluster



For each neighbor that is a core point, recursively add its neighbors

Distance Metric

Typically uses Euclidean distance: d(p,q)=∑i=1n(pi−qi)2d(p,q)=∑i=1n​(pi​−qi​)2​

Advantages





Does not require specifying number of clusters



Can find arbitrarily shaped clusters



Robust to outliers



Only two parameters: ϵϵ and min_samples

Time Complexity





Without spatial indexing: O(n2)O(n2)



With spatial indexing (e.g., KD-tree): O(nlog⁡n)O(nlogn)




"""
import numpy as np

def dbscan(X: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    """
    Implement DBSCAN clustering algorithm.
    
    Args:
        X: Data points of shape (n_samples, n_features)
        eps: Maximum distance between neighbors
        min_samples: Minimum points to form a dense region
    
    Returns:
        Cluster labels where -1 indicates noise
    """
    n_samples = len(X)
    
    # Initialize all points as unvisited (label = -2)
    # -2: unvisited, -1: noise, 0+: cluster ID
    labels = np.full(n_samples, -2, dtype=int)
    
    # Current cluster ID
    cluster_id = 0
    
    # Helper function to compute Euclidean distance
    def euclidean_distance(p1, p2):
        return np.sqrt(np.sum((p1 - p2) ** 2))
    
    # Helper function to find all neighbors within eps
    def get_neighbors(point_idx):
        neighbors = []
        for i in range(n_samples):
            if euclidean_distance(X[point_idx], X[i]) <= eps:
                neighbors.append(i)
        return neighbors
    
    # Helper function to expand cluster
    def expand_cluster(point_idx, neighbors, cluster_id):
        # Assign current point to cluster
        labels[point_idx] = cluster_id
        
        # Use a queue to process neighbors (BFS approach)
        i = 0
        while i < len(neighbors):
            neighbor_idx = neighbors[i]
            
            # If neighbor was marked as noise, add to cluster
            if labels[neighbor_idx] == -1:
                labels[neighbor_idx] = cluster_id
            
            # If neighbor is unvisited
            elif labels[neighbor_idx] == -2:
                labels[neighbor_idx] = cluster_id
                
                # Find neighbors of this neighbor
                neighbor_neighbors = get_neighbors(neighbor_idx)
                
                # If it's a core point, add its neighbors to the queue
                if len(neighbor_neighbors) >= min_samples:
                    # Add new neighbors to the list (avoid duplicates)
                    for nn in neighbor_neighbors:
                        if nn not in neighbors:
                            neighbors.append(nn)
            
            i += 1
    
    # Main DBSCAN algorithm
    for point_idx in range(n_samples):
        # Skip if already processed
        if labels[point_idx] != -2:
            continue
        
        # Find neighbors
        neighbors = get_neighbors(point_idx)
        
        # Check if it's a core point
        if len(neighbors) < min_samples:
            # Mark as noise (may be changed later if it's a border point)
            labels[point_idx] = -1
        else:
            # Start a new cluster
            expand_cluster(point_idx, neighbors, cluster_id)
            cluster_id += 1
    
    return labels


