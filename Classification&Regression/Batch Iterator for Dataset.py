"""
Implement a batch iterable function that samples in a numpy array X and an optional numpy array y. The function should yield batches of a specified size. If y is provided, the function should yield batches of (X, y) pairs; otherwise, it should yield batches of X only.

Example:
Input:
X = np.array([[1, 2], 
                  [3, 4], 
                  [5, 6], 
                  [7, 8], 
                  [9, 10]])
    y = np.array([1, 2, 3, 4, 5])
    batch_size = 2
    batch_iterator(X, y, batch_size)
Output:
[[[[1, 2], [3, 4]], [1, 2]],
     [[[5, 6], [7, 8]], [3, 4]],
     [[[9, 10]], [5]]]
Reasoning:
The dataset X contains 5 samples, and we are using a batch size of 2. Therefore, the function will divide the dataset into 3 batches. The first two batches will contain 2 samples each, and the last batch will contain the remaining sample. The corresponding values from y are also included in each batch.


Understanding Batch Iteration
Batch iteration is a common technique used in machine learning and data processing to handle large datasets more efficiently. Instead of processing the entire dataset at once, which can be memory-intensive, data is processed in smaller, more manageable batches.

Step-by-Step Method to Create a Batch Iterator
Determine the Number of Samples
Calculate the total number of samples in the dataset.

Iterate in Batches
Loop through the dataset in increments of the specified batch size.

Yield Batches
For each iteration, yield a batch of samples from ( X ) and, if provided, the corresponding samples from ( y ).

Key Point
This method ensures efficient processing and can be used for both the training and evaluation phases in machine learning workflows.
"""




import numpy as np

def batch_iterator(X, y=None, batch_size=64):
    n_samples = X.shape[0]
    batches = []
    for i in np.arange(0, n_samples, batch_size):
        begin, end = i, min(i+batch_size, n_samples)
        if y is not None:
            batches.append([X[begin:end], y[begin:end]])
        else:
            batches.append( X[begin:end])
    return batches
    
