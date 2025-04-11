"""
In this problem, you need to implement a 2D convolutional layer in Python. This function will process an input matrix using a specified convolutional kernel, padding, and stride.

Example:
Input:
import numpy as np

input_matrix = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16]
])

kernel = np.array([
    [1, 0],
    [-1, 1]
])

padding = 1
stride = 2

output = simple_conv2d(input_matrix, kernel, padding, stride)
print(output)
Output:
[[ 1.  1. -4.],[ 9.  7. -4.],[ 0. 14. 16.]]
Reasoning:
The function performs a 2D convolution operation on the input matrix using the specified kernel, padding, and stride. The output matrix contains the results of the convolution operation.

"""
"""
Simple Convolutional 2D Layer
The Convolutional layer is a fundamental component used extensively in Computer Vision tasks. Here are the crucial parameters:

Parameters
input_matrix:
A 2D NumPy array representing the input data, such as an image. Each element in this array corresponds to a pixel or a feature value in the input space. The dimensions of the input matrix are typically represented as 
height
×
width
height×width.

kernel:
Another 2D NumPy array representing the convolutional filter. The kernel is smaller than the input matrix and slides over it to perform the convolution operation. Each element in the kernel serves as a weight that modifies the input during convolution. The kernel size is denoted as 
kernel_height
×
kernel_width
kernel_height×kernel_width.

padding:
An integer specifying the number of rows and columns of zeros added around the input matrix. Padding controls the spatial dimensions of the output, allowing the kernel to process edge elements effectively or to maintain the original input size.

stride:
An integer that represents the number of steps the kernel moves across the input matrix for each convolution. A stride greater than one reduces the output size, as the kernel skips over elements.

Implementation
Padding the Input:
The input matrix is padded with zeros based on the specified padding value. This increases the input size and enables the kernel to cover elements at the borders and corners.

Calculating Output Dimensions:
The height and width of the output matrix are calculated using the following formulas:

output_height
=
(
input_height, padded
−
kernel_height
stride
)
+
1
output_height=( 
stride
input_height, padded−kernel_height
​
 )+1
output_width
=
(
input_width, padded
−
kernel_width
stride
)
+
1
output_width=( 
stride
input_width, padded−kernel_width
​
 )+1
Performing Convolution:

A nested loop iterates over each position where the kernel can be applied to the padded input matrix.
At each position, a region of the input matrix, matching the size of the kernel, is selected.
Element-wise multiplication between the kernel and the input region is performed, followed by summing the results to produce a single value. This value is then stored in the corresponding position of the output matrix.
Output:
The function returns the output matrix, which contains the results of the convolution operation performed across the entire input.

"""

import numpy as np

def simple_conv2d(input_matrix: np.ndarray, kernel: np.ndarray, padding: int, stride: int):
    input_height, input_width = input_matrix.shape
    kernel_height, kernel_width = kernel.shape

    padded_input = np.pad(input_matrix, ((padding, padding), (padding, padding)), mode='constant')
    input_height_padded, input_width_padded = padded_input.shape

    output_height = (input_height_padded - kernel_height) // stride + 1
    output_width = (input_width_padded - kernel_width) // stride + 1

    output_matrix = np.zeros((output_height, output_width))

    for i in range(output_height):
        for j in range(output_width):
            region = padded_input[i*stride:i*stride + kernel_height, j*stride:j*stride + kernel_width]
            output_matrix[i, j] = np.sum(region * kernel)

    return output_matrix
