"""
Write a Python function to generate random subsets of a given dataset. The function should take in a 2D numpy array X, a 1D numpy array y, an integer n_subsets, and a boolean replacements. It should return a list of n_subsets random subsets of the dataset, where each subset is a tuple of (X_subset, y_subset). If replacements is True, the subsets should be created with replacements; otherwise, without replacements.

Example:
Input:
X = np.array([[1, 2],
                  [3, 4],
                  [5, 6],
                  [7, 8],
                  [9, 10]])
    y = np.array([1, 2, 3, 4, 5])
    n_subsets = 3
    replacements = False
    get_random_subsets(X, y, n_subsets, replacements)
Output:
[array([[7, 8],
            [1, 2]]), 
     array([4, 1])]
     
    [array([[9, 10],
            [5, 6]]), 
     array([5, 3])]
     
    [array([[3, 4],
            [5, 6]]), 
     array([2, 3])]
Reasoning:
The function generates three random subsets of the dataset without replacements. Each subset includes 50% of the samples (since replacements=False). The samples are randomly selected without duplication.

Understanding Random Subsets of a Dataset
Generating random subsets of a dataset is a useful technique in machine learning, particularly in ensemble methods like bagging and random forests. By creating random subsets, models can be trained on different parts of the data, which helps in reducing overfitting and improving generalization.

Problem Overview
In this problem, you will write a function to generate random subsets of a given dataset. Specifically:

Given a 2D numpy array 
X
X, a 1D numpy array 
y
y, an integer n_subsets, and a boolean replacements, the function will create a list of n_subsets random subsets.
Each subset will be a tuple of 
(
X
subset
,
y
subset
)
(X 
subset
​
 ,y 
subset
​
 ).
Parameters
X
X: A 2D numpy array representing the features.
y
y: A 1D numpy array representing the labels.
n
subsets
n 
subsets
​
 : The number of random subsets to generate.
replacements: A boolean indicating whether to sample with or without replacement.
If replacements is True, subsets will be created with replacements, meaning samples can be repeated within a subset.
If replacements is False, subsets will be created without replacements, meaning samples cannot be repeated within a subset.
Importance
By understanding and implementing this technique, you can enhance the performance of your models through methods like bootstrapping and ensemble learning.
"""

import numpy as np

def get_random_subsets(X, y, n_subsets, replacements=True, seed=42):
	# Your code here
    np.random.seed(seed)
	subset = []
    n_samples = len(X) if replacements else len(X) // 2 
    for _ in range(n_subsets):
        ind = np.random.choice(len(X),size=n_samples,replace = replacements)
        X_subset = X[ind]
        y_subset = y[ind]
        subset.append((X_subset,y_subset))
    return subset
