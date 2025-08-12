"""
Write a Python function that takes a 2-D NumPy array X and an integer degree, generates all polynomial feature combinations of the columns of X up to the given degree inclusive, then sorts the resulting features for each sample from lowest to highest value. The function should return a new 2-D NumPy array whose rows correspond to the input samples and whose columns are the ascending-sorted polynomial features.

Example:
Input:
X = np.array([[2, 3],
              [3, 4],
              [5, 6]])
degree = 2
output = polynomial_features(X, degree)
print(output)
Output:
[[ 1.  2.  3.  4.  6.  9.]
 [ 1.  3.  4.  9. 12. 16.]
 [ 1.  5.  6. 25. 30. 36.]]
Reasoning:
For degree = 2, the raw polynomial terms for the first sample are [1, 2, 3, 4, 6, 9]. Sorting them from smallest to largest yields [1, 2, 3, 4, 6, 9]. The same procedure is applied to every sample.
Understanding Polynomial Features
Generating polynomial features is a method used to create new features for a machine-learning model by raising existing features to a specified power. This technique helps capture non-linear relationships between features.

Example
Given a dataset with two features 
x
1
x 
1
​
  and 
x
2
x 
2
​
 , generating polynomial features up to degree 2 will create new features such as:

x
1
2
x 
1
2
​
 
x
2
2
x 
2
2
​
 
x
1
x
2
x 
1
​
 x 
2
​
 
Problem Overview
In this problem you will write a function to generate polynomial features and then sort each sample's features in ascending order. Specifically:

Given a 2-D NumPy array X and an integer degree, create a new 2-D array with all polynomial combinations of the features up to the specified degree.
Finally, sort each row from the lowest value to the highest value.
Importance
Polynomial expansion allows otherwise linear models to handle non-linear data. Sorting the expanded features can be useful for certain downstream tasks (e.g., histogram-based models or feature selection heuristics) and reinforces array-manipulation skills in NumPy.

"""
import numpy as np
from itertools import combinations_with_replacement

def polynomial_features(X, degree):
    n_samples, n_features = X.shape
    # Generate all combinations of feature indices with replacement up to the given degree
    combs = []
    for d in range(degree + 1):
        combs.extend(combinations_with_replacement(range(n_features), d))
    
    # For each sample, compute the polynomial features
    poly_features = []
    for sample in X:
        features = []
        for comb in combs:
            val = 1
            for idx in comb:
                val *= sample[idx]
            features.append(val)
        # Sort the features for the sample
        poly_features.append(sorted(features))
    
    return np.array(poly_features , dtype = np.float64)
