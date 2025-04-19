"""
Write a Python function to perform a random shuffle of the samples in two numpy arrays, X and y, while maintaining the corresponding order between them. The function should have an optional seed parameter for reproducibility.

Example:
Input:
X = np.array([[1, 2], 
                  [3, 4], 
                  [5, 6], 
                  [7, 8]])
    y = np.array([1, 2, 3, 4])
Output:
(array([[5, 6],
                    [1, 2],
                    [7, 8],
                    [3, 4]]), 
             array([3, 1, 4, 2]))
Reasoning:
The samples in X and y are shuffled randomly, maintaining the correspondence between the samples in both arrays.

"""
"""
Understanding Dataset Shuffling
Random shuffling of a dataset is a common preprocessing step in machine learning to ensure that the data is randomly distributed before training a model. This helps to avoid any potential biases that may arise from the order in which data is presented to the model.

Step-by-Step Method to Shuffle a Dataset
Generate a Random Index Array
Create an array of indices corresponding to the number of samples in the dataset.

Shuffle the Indices
Use a random number generator to shuffle the array of indices.

Reorder the Dataset
Use the shuffled indices to reorder the samples in both ( X ) and ( y ).

Key Point
This method ensures that the correspondence between ( X ) and ( y ) is maintained after shuffling, preserving the relationship between features and labels.

"""
import numpy as np

def shuffle_data(X, y, seed=None):
	# Your code here
	if seed : 	
		np.random.seed(seed)
	ind = np.arange(len(X))
	np.random.shuffle(ind)
	x_shuffle = X[ind]
	y_shuffle = y[ind]
	return x_shuffle,y_shuffle
