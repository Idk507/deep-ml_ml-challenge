"""
Task: Implement Cosine Similarity
In this task, you need to implement a function cosine_similarity(v1, v2) that calculates the cosine similarity between two vectors. Cosine similarity measures the cosine of the angle between two vectors, indicating their directional similarity.

Input:
v1 and v2: Numpy arrays representing the input vectors.
Output:
A float representing the cosine similarity, rounded to three decimal places.
Constraints:
Both input vectors must have the same shape.
Input vectors cannot be empty or have zero magnitude.
Example:
Input:
import numpy as np

v1 = np.array([1, 2, 3])
v2 = np.array([2, 4, 6])
print(cosine_similarity(v1, v2))
Output:
1.0
Reasoning:
The cosine similarity between v1 and v2 is 1.0, indicating perfect similarity.

"""


import numpy as np
from numpy.linalg import norm
def cosine_similarity(v1, v2):
	# Implement your code here
	return np.round(np.dot(v1,v2)/(norm(v1)*norm(v2)),3)
