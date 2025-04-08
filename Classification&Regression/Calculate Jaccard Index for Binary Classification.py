"""
Task: Implement the Jaccard Index
Your task is to implement a function jaccard_index(y_true, y_pred) that calculates the Jaccard Index, a measure of similarity between two binary sets. The Jaccard Index is widely used in binary classification tasks to evaluate the overlap between predicted and true labels.

Your Task:
Implement the function jaccard_index(y_true, y_pred) to:

Calculate the Jaccard Index between the arrays y_true and y_pred.
Return the Jaccard Index as a float value.
Ensure the function handles cases where:
There is no overlap between y_true and y_pred.
Both arrays contain only zeros (edge cases).
The Jaccard Index is defined as:

Jaccard Index
=
Number of elements in the intersection of 
y
true
 and 
y
pred
Number of elements in the union of 
y
true
 and 
y
pred
Jaccard Index= 
Number of elements in the union of y 
true
​
  and y 
pred
​
 
Number of elements in the intersection of y 
true
​
  and y 
pred
​
 
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
y_true = np.array([1, 0, 1, 1, 0, 1])
y_pred = np.array([1, 0, 1, 0, 0, 1])
print(jaccard_index(y_true, y_pred))
Output:
0.75
Reasoning:
The Jaccard Index is calculated as 3 / 4 = 0.75, indicating a 75% overlap between the true and predicted labels.

"""
"""Understanding Jaccard Index in Classification
The Jaccard Index, also known as the Jaccard Similarity Coefficient, is a statistic used to measure the similarity between sets. In the context of binary classification, it measures the overlap between predicted and actual positive labels.

Mathematical Definition
The Jaccard Index is defined as the size of the intersection divided by the size of the union of two sets:

Jaccard Index
=
∣
A
∩
B
∣
∣
A
∪
B
∣
=
∣
A
∩
B
∣
∣
A
∣
+
∣
B
∣
−
∣
A
∩
B
∣
Jaccard Index= 
∣A∪B∣
∣A∩B∣
​
 = 
∣A∣+∣B∣−∣A∩B∣
∣A∩B∣
​
 
In the Context of Binary Classification
Intersection (
A
∩
B
A∩B): The number of positions where both the predicted and true labels are 1 (True Positives).
Union (
A
∪
B
A∪B): The number of positions where either the predicted or true labels (or both) are 1.
Key Properties
Range: The Jaccard Index always falls between 0 and 1 (inclusive).
Perfect Match: A value of 1 indicates identical sets.
No Overlap: A value of 0 indicates disjoint sets.
Symmetry: The index is symmetric, meaning 
J
(
A
,
B
)
=
J
(
B
,
A
)
J(A,B)=J(B,A).
Example
Consider two binary vectors:

True labels: [1, 0, 1, 1, 0, 1]
Predicted labels: [1, 0, 1, 0, 0, 1]
In this case:

Intersection (positions where both are 1): 3.
Union (positions where either is 1): 4.
Jaccard Index: 
3
/
4
=
0.75
3/4=0.75.
Usage in Machine Learning
The Jaccard Index is particularly useful in:

Evaluating clustering algorithms.
Comparing binary classification results.
Document similarity analysis.
Image segmentation evaluation.
When implementing the Jaccard Index, it's important to handle edge cases, such as when both sets are empty (in which case the index is typically defined as 0).

"""

import numpy as np

def jaccard_index(y_true, y_pred):
	# Write your code here
    union = np.sum((y_true ==1) & (y_pred == 1))
    intersect = np.sum((y_true == 1) | (y_pred == 1))
    result = union / intersect
	return round(result, 3)
