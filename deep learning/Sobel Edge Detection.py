"""
Task: Sobel Edge Detection
Edge detection is a fundamental technique in computer vision used to identify boundaries within images. In this task, you will implement a function sobel_edge_detection(image) that applies the Sobel operator to detect edges in a grayscale image.

Input:
image: A 2D list/array representing a grayscale image with pixel values in range [0, 255]
Output:
Return the edge magnitude image as a 2D list with integer values normalized to [0, 255]
The output dimensions will be (H-2, W-2) due to valid convolution (no padding)
Return -1 for invalid inputs
Edge Cases to Handle:
Input is not a valid 2D array
Image dimensions are smaller than 3x3 (minimum required for Sobel)
Any pixel values are outside the valid range (0-255)
Empty image
Notes:
Use the standard Sobel kernels for gradient computation
Compute gradient magnitude from horizontal and vertical gradients
Normalize the output to the range [0, 255] based on the maximum magnitude value
Example:
Input:
image = [[0, 0, 255], [0, 0, 255], [0, 0, 255]]
print(sobel_edge_detection(image))
Output:
[[255]]
Reasoning:
This 3x3 image has a strong vertical edge (black on left, white on right). The Sobel operator computes: Gx = 1020 (strong horizontal gradient), Gy = 0 (no vertical gradient). The magnitude sqrt(1020^2 + 0^2) = 1020 is normalized to 255 since it's the maximum value.

Learn About topic
Sobel Edge Detection
The Sobel operator is a discrete differentiation operator used in image processing and computer vision to detect edges. It computes an approximation of the gradient of the image intensity function.

Sobel Kernels
The Sobel operator uses two 3x3 convolution kernels to compute horizontal and vertical gradients:

Horizontal gradient kernel (Gx): 
G
x
=
[
−
1
0
1
−
2
0
2
−
1
0
1
]
G 
x
​
 = 
​
  
−1
−2
−1
​
  
0
0
0
​
  
1
2
1
​
  
​
 

Vertical gradient kernel (Gy): 
G
y
=
[
−
1
−
2
−
1
0
0
0
1
2
1
]
G 
y
​
 = 
​
  
−1
0
1
​
  
−2
0
2
​
  
−1
0
1
​
  
​
 

Convolution Operation
For each pixel position 
(
i
,
j
)
(i,j), we extract a 3x3 window and compute the element-wise product with each kernel, then sum the results:

G
x
(
i
,
j
)
=
∑
m
=
−
1
1
∑
n
=
−
1
1
I
(
i
+
m
,
j
+
n
)
⋅
K
x
(
m
+
1
,
n
+
1
)
G 
x
​
 (i,j)=∑ 
m=−1
1
​
 ∑ 
n=−1
1
​
 I(i+m,j+n)⋅K 
x
​
 (m+1,n+1)

G
y
(
i
,
j
)
=
∑
m
=
−
1
1
∑
n
=
−
1
1
I
(
i
+
m
,
j
+
n
)
⋅
K
y
(
m
+
1
,
n
+
1
)
G 
y
​
 (i,j)=∑ 
m=−1
1
​
 ∑ 
n=−1
1
​
 I(i+m,j+n)⋅K 
y
​
 (m+1,n+1)

where 
I
I is the input image and 
K
x
K 
x
​
 , 
K
y
K 
y
​
  are the Sobel kernels.

Gradient Magnitude
The edge strength at each pixel is computed as the magnitude of the gradient vector:

G
=
G
x
2
+
G
y
2
G= 
G 
x
2
​
 +G 
y
2
​
 
​
 

This gives the rate of change of intensity regardless of direction. Strong edges produce high magnitude values.

Gradient Direction
The direction of the edge can also be computed (though not required for this task):

θ
=
arctan
⁡
(
G
y
G
x
)
θ=arctan( 
G 
x
​
 
G 
y
​
 
​
 )

Normalization
To display the result as an image, we normalize the magnitude values to the range 
[
0
,
255
]
[0,255]:

G
n
o
r
m
a
l
i
z
e
d
=
G
G
m
a
x
×
255
G 
normalized
​
 = 
G 
max
​
 
G
​
 ×255

Why Sobel Works
The Sobel kernels are designed to:

Detect changes: The +1 and -1 values compute differences between neighboring pixels
Smooth noise: The center row/column weights (2, 0, 2) provide Gaussian-like smoothing
Separate directions: 
G
x
G 
x
​
  detects vertical edges, 
G
y
G 
y
​
  detects horizontal edges
Applications
Object detection and recognition
Image segmentation
Feature extraction for machine learning
Preprocessing for more complex edge detectors (Canny)


""""
import numpy as np

def sobel_edge_detection(image):
    """
    Apply Sobel edge detection to a grayscale image.

    Args:
        image: 2D list/array representing a grayscale image
               with values in range [0, 255]

    Returns:
        Edge magnitude image as 2D list with integer values (0-255),
        or -1 if input is invalid
    """

    # --- Input Validation ---
    if not isinstance(image, (list, np.ndarray)):
        return -1
    try:
        arr = np.array(image, dtype=np.float64)
    except:
        return -1

    # Must be 2D
    if arr.ndim != 2:
        return -1

    H, W = arr.shape
    # Minimum size requirement
    if H < 3 or W < 3:
        return -1

    # Pixel range check
    if np.any(arr < 0) or np.any(arr > 255):
        return -1

    # --- Sobel Kernels ---
    Gx_kernel = np.array([[-1, 0, 1],
                          [-2, 0, 2],
                          [-1, 0, 1]])
    Gy_kernel = np.array([[-1, -2, -1],
                          [ 0,  0,  0],
                          [ 1,  2,  1]])

    # --- Convolution (valid mode: no padding) ---
    out_H, out_W = H - 2, W - 2
    magnitude = np.zeros((out_H, out_W), dtype=np.float64)

    for i in range(out_H):
        for j in range(out_W):
            region = arr[i:i+3, j:j+3]
            Gx = np.sum(region * Gx_kernel)
            Gy = np.sum(region * Gy_kernel)
            magnitude[i, j] = np.sqrt(Gx**2 + Gy**2)

    # --- Normalization to [0, 255] ---
    max_val = magnitude.max()
    if max_val == 0:
        norm = np.zeros_like(magnitude, dtype=np.uint8)
    else:
        norm = (magnitude / max_val) * 255
        norm = norm.astype(np.uint8)

    return norm.tolist()

