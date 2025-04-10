"""
K-Means Clustering

Your task is to write a Python function that implements the k-Means clustering algorithm. This function should take specific inputs and produce a list of final centroids. k-Means clustering is a method used to partition n points into k clusters. The goal is to group similar points together and represent each group by its center (called the centroid).

Function Inputs:
points: A list of points, where each point is a tuple of coordinates (e.g., (x, y) for 2D points)
k: An integer representing the number of clusters to form
initial_centroids: A list of initial centroid points, each a tuple of coordinates
max_iterations: An integer representing the maximum number of iterations to perform
Function Output:
A list of the final centroids of the clusters, where each centroid is rounded to the nearest fourth decimal.

Example:
Input:
points = [(1, 2), (1, 4), (1, 0), (10, 2), (10, 4), (10, 0)], k = 2, initial_centroids = [(1, 1), (10, 1)], max_iterations = 10
Output:
[(1, 2), (10, 2)]
Reasoning:
Given the initial centroids and a maximum of 10 iterations, the points are clustered around these points, and the centroids are updated to the mean of the assigned points, resulting in the final centroids which approximate the means of the two clusters. The exact number of iterations needed may vary, but the process will stop after 10 iterations at most.

"""
"""
K-Means Clustering Algorithm Implementation
Algorithm Steps
Initialization
Use the provided initial_centroids as your starting point. This step is already done for you in the input.

Assignment Step
For each point in your dataset:

Calculate its distance to each centroid.
Assign the point to the cluster of the nearest centroid.
Hint: Consider creating a helper function to calculate the Euclidean distance between two points.
Update Step
For each cluster:

Calculate the mean of all points assigned to the cluster.
Update the centroid to this new mean position.
Hint: Be careful with potential empty clusters. Decide how you'll handle them (e.g., keep the previous centroid).
Iteration
Repeat steps 2 and 3 until either:

The centroids no longer change significantly (this case does not need to be included in your solution), or
You reach the max_iterations limit.
Hint: You might want to keep track of the previous centroids to check for significant changes.
Result
Return the list of final centroids, ensuring each coordinate is rounded to the nearest fourth decimal.

"""

import numpy as np

def euclidean_distance(a, b):
    return np.sqrt(((a - b) ** 2).sum(axis=1))

def k_means_clustering(points, k, initial_centroids, max_iterations):
    points = np.array(points)
    centroids = np.array(initial_centroids)
    
    for iteration in range(max_iterations):
        # Assign points to the nearest centroid
        distances = np.array([euclidean_distance(points, centroid) for centroid in centroids])
        assignments = np.argmin(distances, axis=0)

        new_centroids = np.array([points[assignments == i].mean(axis=0) if len(points[assignments == i]) > 0 else centroids[i] for i in range(k)])
        
        # Check for convergence
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids
        centroids = np.round(centroids,4)
    return [tuple(centroid) for centroid in centroids]
