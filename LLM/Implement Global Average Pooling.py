"""
Implement a function that performs Global Average Pooling on a 3D NumPy array representing feature maps from a convolutional layer. The function should take an input of shape (height, width, channels) and return a 1D array of shape (channels,), where each element is the average of all values in the corresponding feature map.

Example:
Input:
x = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
Output:
[5.5, 6.5, 7.5]
Reasoning:
For each channel, compute the average of all elements. For channel 0: (1+4+7+10)/4 = 5.5, for channel 1: (2+5+8+11)/4 = 6.5, for channel 2: (3+6+9+12)/4 = 7.5.

"""
import numpy as np

def global_avg_pool(x: np.ndarray) -> np.ndarray:
	# Your code here
	return np.mean(x,axis=(0,1))

import numpy as np

def global_avg_pool(x: np.ndarray) -> np.ndarray:
	# Your code here
	


    H, W, C = x.shape
    output = np.zeros(C)
    
    for c in range(C):
        sum_c = 0.0
        for i in range(H):
            for j in range(W):
                sum_c += x[i, j, c]
        output[c] = sum_c / (H * W)
    
    return output


