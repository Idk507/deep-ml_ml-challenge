"""

Task: Generate a Confusion Matrix
Your task is to implement the function confusion_matrix(data) that generates a confusion matrix for a binary classification problem. The confusion matrix provides a summary of the prediction results on a classification problem, allowing you to visualize how many data points were correctly or incorrectly labeled.

Input:
A list of lists, where each inner list represents a pair
[y_true, y_pred] for one observation. y_true is the actual label, and y_pred is the predicted label.
Output:
A 
2
×
2
2×2 confusion matrix represented as a list of lists.
Example:
Input:
data = [[1, 1], [1, 0], [0, 1], [0, 0], [0, 1]]
print(confusion_matrix(data))
Output:
[[1, 1], [2, 1]]
Reasoning:
The confusion matrix shows the counts of true positives, false negatives, false positives, and true negatives.

"""

"""
Understanding Orthogonal Projection in Vector Spaces
Orthogonal projection is a fundamental concept in linear algebra, used to project one vector onto another. The projection of vector 
v
v onto a line defined by vector 
L
L results in a new vector that lies on 
L
L, representing the closest point to 
v
v on that line. This can be thought of as 
v
v's shadow on 
L
L if a light was shown directly down on 
v
v.

To project a vector 
v
v onto a non-zero vector 
L
L in space, we calculate the scalar projection of 
v
v onto the unit vector of 
L
L, which represents the magnitude of the projection. The resulting projection vector lies along the direction of 
L
L.

For any vector 
v
v in Cartesian space, the orthogonal projection onto 
L
L is calculated using the formula:

proj
L
(
v
)
=
v
⋅
L
L
⋅
L
L
proj 
L
​
 (v)= 
L⋅L
v⋅L
​
 L
Where:

v
v is the vector being projected,
L
L is the vector defining the line of projection,
v
⋅
L
v⋅L is the dot product of 
v
v and 
L
L,
L
⋅
L
L⋅L is the dot product of 
L
L with itself, which gives the magnitude squared of 
L
L.
The resulting projection vector lies along the direction of 
L
L and represents the component of 
v
v that is parallel to 
L
L.

More generally, the projection of 
v
v onto a unit vector 
L
^
L
^
  (the normalized version of 
L
L) simplifies to:

proj
L
(
v
)
=
(
v
⋅
L
^
)
L
^
proj 
L
​
 (v)=(v⋅ 
L
^
 ) 
L
^
 
Applications of Orthogonal Projection
Orthogonal projection has a wide range of applications across various fields in mathematics, physics, computer science, and engineering. Some of the most common applications include:

Computer Graphics: In 3D rendering, orthogonal projections are used to create 2D views of 3D objects. This projection helps in reducing dimensional complexity and displaying models from different angles.
Data Science and Machine Learning: In high-dimensional data, projection methods are used to reduce dimensions (e.g., Principal Component Analysis) by projecting data onto lower-dimensional subspaces, helping with data visualization and reducing computational complexity.

"""

from collections import Counter

def confusion_matrix(data):
    # Count all occurrences
    counts = Counter(tuple(pair) for pair in data)
    # Get metrics
    TP, FN, FP, TN = counts[(1, 1)], counts[(1, 0)], counts[(0, 1)], counts[(0, 0)]
    # Define matrix and return
    confusion_matrix = [[TP, FN], [FP, TN]]
    return confusion_matrix
