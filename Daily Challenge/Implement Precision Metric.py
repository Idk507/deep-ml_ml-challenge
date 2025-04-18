"""
Write a Python function precision that calculates the precision metric given two numpy arrays: y_true and y_pred. The y_true array contains the true binary labels, and the y_pred array contains the predicted binary labels. Precision is defined as the ratio of true positives to the sum of true positives and false positives.

Example:
Input:
import numpy as np

y_true = np.array([1, 0, 1, 1, 0, 1])
y_pred = np.array([1, 0, 1, 0, 0, 1])

result = precision(y_true, y_pred)
print(result)
Output:
1.0
Reasoning:
True Positives (TP) = 3
False Positives (FP) = 0
Precision = TP / (TP + FP) = 3 / (3 + 0) = 1.0


"""
"""
Understanding Precision in Classification
Precision is a key metric used in the evaluation of classification models, particularly in binary classification. It provides insight into the accuracy of the positive predictions made by the model.

Mathematical Definition
Precision is defined as the ratio of true positives (TP) to the sum of true positives and false positives (FP):

Precision= TP+FP/TP
​
 
Where:

True Positives (TP): The number of positive samples that are correctly identified as positive.
False Positives (FP): The number of negative samples that are incorrectly identified as positive.
Characteristics of Precision
Range: Precision ranges from 0 to 1, where 1 indicates perfect precision (no false positives) and 0 indicates no true positives.
Interpretation: High precision means that the model has a low false positive rate, meaning it rarely labels negative samples as positive.
Use Case: Precision is particularly useful when the cost of false positives is high, such as in medical diagnosis or fraud detection.
In this problem, you will implement a function to calculate precision given the true labels and predicted labels of a binary classification task.

"""
import numpy as np
def precision(y_true, y_pred):
	tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    return tp / (tp+fp)
