"""
Implement Recall Metric in Binary Classification

Task: Implement Recall in Binary Classification
Your task is to implement the recall metric in a binary classification setting. Recall is a performance measure that evaluates how effectively a machine learning model identifies positive instances from all the actual positive cases in a dataset.

You need to write a function recall(y_true, y_pred) that calculates the recall metric. The function should accept two inputs:

y_true: A list of true binary labels (0 or 1) for the dataset.
y_pred: A list of predicted binary labels (0 or 1) from the model.
Your function should return the recall value rounded to three decimal places. If the denominator (TP + FN) is zero, the recall should be 0.0 to avoid division by zero.

Example:
Input:
import numpy as np

y_true = np.array([1, 0, 1, 1, 0, 1])
y_pred = np.array([1, 0, 1, 0, 0, 1])

print(recall(y_true, y_pred))
Output:
# 0.75
Reasoning:
The recall value for the given true labels and predicted labels is 0.75. The model correctly identified 3 out of 4 positive instances in the dataset.

"""
import numpy as np
def recall(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return round(tp /(tp+fn),3)
    
