"""
Write a Python function to perform one-hot encoding of nominal values. The function should take in a 1D numpy array x of integer values and an optional integer n_col representing the number of columns for the one-hot encoded array. If n_col is not provided, it should be automatically determined from the input array.

Example:
Input:
x = np.array([0, 1, 2, 1, 0])
    output = to_categorical(x)
    print(output)
Output:
# [[1. 0. 0.]
    #  [0. 1. 0.]
    #  [0. 0. 1.]
    #  [0. 1. 0.]
    #  [1. 0. 0.]]
Reasoning:
Each element in the input array is transformed into a one-hot encoded vector, where the index corresponding to the value in the input array is set to 1, and all other indices are set to 0.

"""

import numpy as np

def to_categorical(x, n_col=None):
	# Your code here
	if n_col is None :
		n_col = np.max(x)+1 #no. of col 
	one_hot = np.zeros((x.shape[0],n_col))
	one_hot[np.arange(x.shape[0]),x] = 1
	return one_hot
