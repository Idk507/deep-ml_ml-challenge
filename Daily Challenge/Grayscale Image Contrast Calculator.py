"""
Write a Python function to calculate the contrast of a grayscale image using the difference between the maximum and minimum pixel values.

Example:
Input:
img = np.array([[0, 50], [200, 255]])
Output:
255
Reasoning:
The function calculates contrast by finding the difference between the maximum (255) and minimum (0) pixel values in the image, resulting in a contrast of 255.

"""
"""
Calculating Contrast of a Grayscale Image
Contrast in a grayscale image refers to the difference in luminance or color that makes an object distinguishable. Here are methods to calculate contrast:

1. Basic Contrast Calculation
The simplest way to define the contrast of a grayscale image is by using the difference between the maximum and minimum pixel values:

Contrast
=
max
⁡
(
I
)
−
min
⁡
(
I
)
Contrast=max(I)−min(I)
2. RMS Contrast
Root Mean Square (RMS) contrast considers the standard deviation of pixel intensities:

RMS Contrast
=
σ
μ
RMS Contrast= 
μ
σ
​
 
3. Michelson Contrast
Michelson contrast is defined as:

C
=
I
max
−
I
min
I
max
+
I
min
C= 
I 
max
​
 +I 
min
​
 
I 
max
​
 −I 
min
​
 
​
 
Example Calculation
For a grayscale image with pixel values ranging from 50 to 200:

Maximum Pixel Value: 200
Minimum Pixel Value: 50
Contrast Calculation:
Contrast
=
200
−
50
=
150
Contrast=200−50=150
Applications
Calculating contrast is crucial in:

Image quality assessment
Preprocessing in computer vision
Enhancing visibility in images
Object detection and analysis

"""
import numpy as np

def calculate_contrast(img):
    """
    Calculate the contrast of a grayscale image.
    Args:
        img (numpy.ndarray): 2D array representing a grayscale image with pixel values between 0 and 255.
    Returns:
        float: Contrast value rounded to 3 decimal places.
    """
    # Find the maximum and minimum pixel values
    max_pixel = np.max(img)
    min_pixel = np.min(img)

    # Calculate contrast
    contrast = max_pixel - min_pixel

    return round(float(contrast), 3)
