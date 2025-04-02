"""
Calculate Accuracy Score

Write a Python function to calculate the accuracy score of a model's predictions. The function should take in two 1D numpy arrays: y_true, which contains the true labels, and y_pred, which contains the predicted labels. It should return the accuracy score as a float.

Example:
Input:
y_true = np.array([1, 0, 1, 1, 0, 1])
    y_pred = np.array([1, 0, 0, 1, 0, 1])
    output = accuracy_score(y_true, y_pred)
    print(output)
Output:
# 0.8333333333333334
Reasoning:
The function compares the true labels with the predicted labels and calculates the ratio of correct predictions to the total number of predictions. In this example, there are 5 correct predictions out of 6, resulting in an accuracy score of 0.8333333333333334.

"""
"""
Understanding Accuracy Score
Accuracy is a metric used to evaluate the performance of a classification model. It is defined as the ratio of the number of correct predictions to the total number of predictions made. Mathematically, accuracy is given by:

Accuracy
=
Number of Correct Predictions
Total Number of Predictions
Accuracy= 
Total Number of Predictions
Number of Correct Predictions
​
 
Problem Overview
In this problem, you will write a function to calculate the accuracy score given the true labels and the predicted labels. The function will compare the two arrays and compute the accuracy as the proportion of matching elements.

Importance
Accuracy is a straightforward and commonly used metric for classification tasks. It provides a quick way to understand how well a model is performing, but it may not always be the best metric, especially for imbalanced datasets.

"""

import numpy as np

def accuracy_score(y_true, y_pred):
	acc = np.sum(y_pred == y_true,axis=0) /len(y_true)
    return acc
