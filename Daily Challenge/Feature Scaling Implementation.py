
"""
Feature Scaling Implementation

Write a Python function that performs feature scaling on a dataset using both standardization and min-max normalization. The function should take a 2D NumPy array as input, where each row represents a data sample and each column represents a feature. It should return two 2D NumPy arrays: one scaled by standardization and one by min-max normalization. Make sure all results are rounded to the nearest 4th decimal.

Example:
Input:
data = np.array([[1, 2], [3, 4], [5, 6]])
Output:
([[-1.2247, -1.2247], [0.0, 0.0], [1.2247, 1.2247]], [[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
Reasoning:
Standardization rescales the feature to have a mean of 0 and a standard deviation of 1. Min-max normalization rescales the feature to a range of [0, 1], where the minimum feature value maps to 0 and the maximum to 1.

Learn About topic
Feature Scaling Techniques
Feature scaling is crucial in many machine learning algorithms that are sensitive to the magnitude of features. This includes algorithms that use distance measures, like k-nearest neighbors, and gradient descent-based algorithms, like linear regression.

Standardization
Standardization (or Z-score normalization) is the process where features are rescaled so that they have the properties of a standard normal distribution with a mean of zero and a standard deviation of one:

z
=
(
x
−
μ
)
σ
z= 
σ
(x−μ)
​
 
where ( x ) is the original feature, ( \mu ) is the mean of that feature, and ( \sigma ) is the standard deviation.

Min-Max Normalization
Min-max normalization rescales the feature to a fixed range, typically 0 to 1, or it can be shifted to any range ([a, b]) by transforming the data using the formula:

x
′
=
(
x
−
min
(
x
)
)
(
max
(
x
)
−
min
(
x
)
)
×
(
max
−
min
)
+
min
x 
′
 = 
(max(x)−min(x))
(x−min(x))
​
 ×(max−min)+min
where ( x ) is the original value, ( \text{min}(x) ) is the minimum value for that feature, ( \text{max}(x) ) is the maximum value, and ( \text{min} ) and ( \text{max} ) are the new minimum and maximum values for the scaled data.

Key Points
Equal Contribution: Implementing these scaling techniques ensures that features contribute equally to the development of the model.
Improved Convergence: Feature scaling can significantly improve the convergence speed of learning algorithms.
This structured explanation outlines the importance of feature scaling and describes two commonly used techniques with their mathematical formulas

"""


import numpy as np

def feature_scaling(data):
    # Standardization
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    standardized_data = (data - mean) / std
    
    # Min-Max Normalization
    min_val = np.min(data, axis=0)
    max_val = np.max(data, axis=0)
    normalized_data = (data - min_val) / (max_val - min_val)
    
    return np.round(standardized_data,4).tolist(), np.round(normalized_data,4).tolist()
