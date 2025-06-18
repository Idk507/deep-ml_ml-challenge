"""
Implement Gini Impurity Calculation for a Set of Classes

Task: Implement Gini Impurity Calculation
Your task is to implement a function that calculates the Gini Impurity for a set of classes. Gini impurity is commonly used in decision tree algorithms to measure the impurity or disorder within a node.

Example:
Input:
y = [0, 1, 1, 1, 0]
print(gini_impurity(y))
Output:
0.48
Reasoning:
The Gini Impurity is calculated as 1 - (p_0^2 + p_1^2), where p_0 and p_1 are the probabilities of each class. In this case, p_0 = 2/5 and p_1 = 3/5, resulting in a Gini Impurity of 0.48.

"""

import numpy as np

def gini_impurity(y):
	"""
	Calculate Gini Impurity for a list of class labels.

	:param y: List of class labels
	:return: Gini Impurity rounded to three decimal places
	"""
	n = len(y)
    cls,cnt = np.unique(y,return_counts = True)
    pro = cnt/cnt.sum()
    gini = 1.0 - np.sum(pro**2)
    return np.round(gini,3)
