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

"""

def calculate_brightness(img):
	# Write your code here
	if not img or not img[0]: return    -1 
    row,col = len(img),len(img[0])

    for r in img :
        if len(r) !=col :
            return -1
        for p in r:
            if not 0 <= p <=255 : return -1
    tool = sum(sum(r)for r in img)
    return round(tool/(row*col),2)
