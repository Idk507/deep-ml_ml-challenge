"""
Apply Zero Padding to an Image
Easy
Computer Vision

Task: Zero Padding for Images
In this task, you will implement a function zero_pad_image(img, pad_width) that adds zero padding around a grayscale image.

Zero padding is a fundamental operation in image processing and convolutional neural networks where layers of zeros are added around the border of an image.

Your Task:
Implement the function zero_pad_image(img, pad_width) to:

Add pad_width rows/columns of zeros on each side of the image (top, bottom, left, right).
Return the padded image as a 2D list with integer values.
Handle edge cases:
If the input is not a valid 2D array.
If the image has empty dimensions.
If pad_width is not a non-negative integer.
For any of these edge cases, the function should return -1.

Example:
Input:
img = [[1, 2], [3, 4]]
pad_width = 1
print(zero_pad_image(img, pad_width))
Output:
[[0, 0, 0, 0], [0, 1, 2, 0], [0, 3, 4, 0], [0, 0, 0, 0]]
Reasoning:
The original 2x2 image gets a border of zeros added on all sides. The new dimensions are (2+21) x (2+21) = 4x4. The original pixel values remain in the center while zeros fill the border.

Learn About topic
Zero Padding in Image Processing
Zero padding is a fundamental operation in image processing and deep learning where we add layers of zeros around the border of an image.

Mathematical Definition
Given an input image 
I
I of dimensions 
H
×
W
H×W and a padding width 
p
p, the padded image 
I
′
I 
′
  has dimensions:

H
′
=
H
+
2
p
H 
′
 =H+2p 
W
′
=
W
+
2
p
W 
′
 =W+2p

The padded image can be expressed as:

I
′
(
i
,
j
)
=
{
I
(
i
−
p
,
j
−
p
)
if 
p
≤
i
<
H
+
p
 and 
p
≤
j
<
W
+
p
0
otherwise
I 
′
 (i,j)={ 
I(i−p,j−p)
0
​
  
if p≤i<H+p and p≤j<W+p
otherwise
​
 

Why Use Zero Padding?
1. Preserve Spatial Dimensions
In convolutional neural networks, convolution operations reduce the spatial dimensions of feature maps. Padding helps maintain the original size through multiple layers.

2. Edge Pixel Importance
Without padding, edge pixels contribute less to the output than center pixels because they participate in fewer convolution operations. Padding ensures more equal treatment of all pixels.

3. Control Output Size
With proper padding, the output of a convolution can match the input size. This is known as "same" padding.

Convolution Output Size Formula
For a convolution with kernel size 
k
k, stride 
s
s, and padding 
p
p, the output dimension is:

O
=
⌊
I
−
k
+
2
p
s
⌋
+
1
O=⌊ 
s
I−k+2p
​
 ⌋+1

where 
I
I is the input dimension and 
O
O is the output dimension.

Types of Padding
Zero Padding: Fill borders with zeros (most common)
Reflect Padding: Mirror the edge values
Replicate Padding: Repeat the edge values
Visual Example
For a 
2
×
2
2×2 image with padding 
p
=
1
p=1:

Original: 
(
1
2
3
4
)
( 
1
3
​
  
2
4
​
 )

After Zero Padding: 
(
0
0
0
0
0
1
2
0
0
3
4
0
0
0
0
0
)
​
  
0
0
0
0
​
  
0
1
3
0
​
  
0
2
4
0
​
  
0
0
0
0
​
  
​
 

The output dimensions are 
(
2
+
2
×
1
)
×
(
2
+
2
×
1
)
=
4
×
4
(2+2×1)×(2+2×1)=4×4.

"""
def zero_pad_image(img, pad_width):
    """
    Add zero padding around a grayscale image.

    Args:
        img: 2D list of integers (grayscale image)
        pad_width: non-negative integer, number of pixels to pad on each side

    Returns:
        Padded image as 2D list with integer values,
        or -1 if input is invalid
    """

    # --- Edge Case Checks ---
    # Check pad_width validity
    if not isinstance(pad_width, int) or pad_width < 0:
        return -1

    # Check img is a non-empty 2D list
    if not isinstance(img, list) or len(img) == 0:
        return -1
    if not all(isinstance(row, list) for row in img):
        return -1
    if not all(len(row) == len(img[0]) for row in img):  # ensure rectangular
        return -1
    if len(img[0]) == 0:  # empty row case
        return -1

    H, W = len(img), len(img[0])  # original dimensions
    H_new, W_new = H + 2 * pad_width, W + 2 * pad_width

    # --- Initialize padded image with zeros ---
    padded_img = [[0 for _ in range(W_new)] for _ in range(H_new)]

    # --- Copy original image into center ---
    for i in range(H):
        for j in range(W):
            padded_img[i + pad_width][j + pad_width] = img[i][j]

    return padded_img
