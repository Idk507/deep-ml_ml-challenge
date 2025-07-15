"""
Implement a function that scans every feature and threshold in a small data set, then returns the split that minimises the weighted Gini impurity. Your implementation should support binary class labels (0 or 1) and handle ties gracefully.

You will write one function:

find_best_split(X: np.ndarray, y: np.ndarray) -> tuple[int, float]
X is an 
n
×
d
n×d NumPy array of numeric features.
y is a length-
n
n NumPy array of 0/1 labels.
The function returns (best_feature_index, best_threshold) for the split with the lowest weighted Gini impurity.
If several splits share the same impurity, return the first that you encounter while scanning features and thresholds.
Example:
Input:
import numpy as np
X = np.array([[2.5],[3.5],[1.0],[4.0]])
y = np.array([0,1,0,1])
print(find_best_split(X, y))
Output:
(0, 2.5)
Reasoning:
Splitting on feature 0 at threshold 2.5 yields two perfectly pure leaves, producing the minimum possible weighted Gini impurity.

"""
import numpy as np
from typing import Tuple

def gini_impurity(y: np.ndarray) -> float:
    if len(y) == 0:
        return 0.0
    p = np.bincount(y, minlength=2) / len(y)
    return 1.0 - np.sum(p ** 2)

def find_best_split(X: np.ndarray, y: np.ndarray) -> Tuple[int, float]:
    n, d = X.shape
    best_gini = float('inf')
    best_split = (-1, float('inf'))

    for feature_idx in range(d):
        thresholds = np.unique(X[:, feature_idx])
        for threshold in thresholds:
            left_mask = X[:, feature_idx] <= threshold
            right_mask = ~left_mask
            y_left = y[left_mask]
            y_right = y[right_mask]

            n_left, n_right = len(y_left), len(y_right)
            if n_left == 0 or n_right == 0:
                continue

            gini_left = gini_impurity(y_left)
            gini_right = gini_impurity(y_right)
            weighted_gini = (n_left / n) * gini_left + (n_right / n) * gini_right

            if weighted_gini < best_gini:
                best_gini = weighted_gini
                best_split = (feature_idx, threshold)

    return best_split
