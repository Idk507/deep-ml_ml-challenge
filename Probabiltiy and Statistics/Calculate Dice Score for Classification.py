"""
Calculate Dice Score for Classification

Task: Compute the Dice Score
Your task is to implement a function dice_score(y_true, y_pred) that calculates the Dice Score, also known as the SÃ¸rensen-Dice coefficient or F1-score, for binary classification. The Dice Score is used to measure the similarity between two sets and is particularly useful in tasks like image segmentation and binary classification.

Your Task:
Implement the function dice_score(y_true, y_pred) to:

Calculate the Dice Score between the arrays y_true and y_pred.
Return the Dice Score as a float value rounded to 3 decimal places.
Handle edge cases appropriately, such as when there are no true or predicted positives.
The Dice Score is defined as:

Dice Score
=
2
×
(
Number of elements in the intersection of 
y
true
 and 
y
pred
)
Number of elements in 
y
true
+
Number of elements in 
y
pred
Dice Score= 
Number of elements in y 
true
​
 +Number of elements in y 
pred
​
 
2×(Number of elements in the intersection of y 
true
​
  and y 
pred
​
 )
​
 
Where:

y
true
y 
true
​
  and 
y
pred
y 
pred
​
  are binary arrays of the same length, representing true and predicted labels.
The result ranges from 0 (no overlap) to 1 (perfect overlap).
Example:
Input:
y_true = np.array([1, 1, 0, 1, 0, 1])
y_pred = np.array([1, 1, 0, 0, 0, 1])
print(dice_score(y_true, y_pred))
Output:
0.857
Reasoning:
The Dice Score is calculated as (2 * 3) / (2 * 3 + 0 + 1) = 0.857, indicating an 85.7% overlap between the true and predicted labels.

"""

import numpy as np

def dice_score(y_true, y_pred):
	# Write your code here
	y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    intersect = np.logical_and(y_true,y_pred).sum()
    total =  y_true.sum() +y_pred.sum()
    if total == 0.0 : 
        return 0.0
    dice = 2*intersect / total
    return round(dice,3)
