"""
Task: Image Brightness Calculator
In this task, you will implement a function calculate_brightness(img) that calculates the average brightness of a grayscale image. The image is represented as a 2D matrix, where each element represents a pixel value between 0 (black) and 255 (white).

Your Task:
Implement the function calculate_brightness(img) to:

Return the average brightness of the image rounded to two decimal places.
Handle edge cases:
If the image matrix is empty.
If the rows in the matrix have inconsistent lengths.
If any pixel values are outside the valid range (0-255).
For any of these edge cases, the function should return -1.

Example:
Input:
img = [
    [100, 200],
    [50, 150]
]
print(calculate_brightness(img))
Output:
125.0
Reasoning:
The average brightness is calculated as (100 + 200 + 50 + 150) / 4 = 125.0

Image Brightness Calculator
Consider a grayscale image represented as a 2D matrix where each element represents a pixel value between 0 (black) and 255 (white):

I
m
a
g
e
=
(
p
11
p
12
p
21
p
22
)
Image=( 
p 
11
​
 
p 
21
​
 
​
  
p 
12
​
 
p 
22
​
 
​
 )
The average brightness is calculated as:

B
r
i
g
h
t
n
e
s
s
=
∑
i
=
1
m
∑
j
=
1
n
p
i
j
m
×
n
Brightness= 
m×n
∑ 
i=1
m
​
 ∑ 
j=1
n
​
 p 
ij
​
 
​
 
Where:

p
i
j
p 
ij
​
  is the pixel value at position 
(
i
,
j
)
(i,j)
m
m is the number of rows
n
n is the number of columns
Things to Note:
All pixel values must be between 0 and 255
The image matrix must be well-formed (all rows same length)
Empty or invalid images return -1
"""

def calculate_brightness(img):
    # Check if image is empty or has no columns
    if not img or not img[0]:
        return -1

    rows, cols = len(img), len(img[0])

    # Check if all rows have same length and values are valid
    for row in img:
        if len(row) != cols:
            return -1
        for pixel in row:
            if not 0 <= pixel <= 255:
                return -1

    # Calculate average brightness
    total = sum(sum(row) for row in img)
    return round(total / (rows * cols), 2)

