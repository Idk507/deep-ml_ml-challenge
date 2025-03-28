"""
One-Hot Encoding of Nominal Values

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

def one_hot_encode(data, num_classes=None):
    # Determine the number of classes if not provided
    if num_classes is None:
        num_classes = np.max(data) + 1
    # Initialize the one-hot encoded array
    one_hot = np.zeros((len(data), num_classes))
    # Set the appropriate index in each row to 1
    one_hot[np.arange(len(data)), data] = 1
    return one_hot

# Example usage:
data = np.array([0, 1, 2, 1, 0])
one_hot_encoded = one_hot_encode(data)
print(one_hot_encoded)
