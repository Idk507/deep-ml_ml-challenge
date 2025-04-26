"""
Task: Implement F-Score Calculation for Binary Classification
Your task is to implement a function that calculates the F-Score for a binary classification task. The F-Score combines both Precision and Recall into a single metric, providing a balanced measure of a model's performance.

Write a function f_score(y_true, y_pred, beta) where:

y_true: A numpy array of true labels (binary).
y_pred: A numpy array of predicted labels (binary).
beta: A float value that adjusts the importance of Precision and Recall. When beta=1, it computes the F1-Score, a balanced measure of both Precision and Recall.
The function should return the F-Score rounded to three decimal places.

Example:
Input:
y_true = np.array([1, 0, 1, 1, 0, 1])
y_pred = np.array([1, 0, 1, 0, 0, 1])
beta = 1

print(f_score(y_true, y_pred, beta))
Output:
0.857
Reasoning:
The F-Score for the binary classification task is calculated using the true labels, predicted labels, and beta value.

Understanding F-Score in Classification
F-Score, also called F-measure, is a measure of predictive performance that's calculated from the Precision and Recall metrics.

Mathematical Definition
The 
F
β
F 
β
​
  score applies additional weights, valuing one of precision or recall more than the other. When 
β
β equals 1, also known as the F1-Score, it symmetrically represents both precision and recall in one metric. The F-Score can be calculated using the following formula:

F
β
=
(
1
+
β
2
)
×
precision
×
recall
(
β
2
×
precision
)
+
recall
F 
β
​
 =(1+β 
2
 )× 
(β 
2
 ×precision)+recall
precision×recall
​
 
Where:

Recall: The number of true positive results divided by the number of all samples that should have been identified as positive.
Precision: The number of true positive results divided by the number of all samples predicted to be positive, including those not identified correctly.
Implementation Instructions
In this problem, you will implement a function to calculate the F-Score given the true labels, predicted labels, and the Beta value of a binary classification task. The results should be rounded to three decimal places.

Special Case:
If the denominator is zero, the F-Score should be set to 0.0 to avoid division by zero.
"""

import numpy as np

def f_score(y_true, y_pred, beta):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0

    op = precision * recall
    div = ((beta**2) * precision) + recall

    if div == 0 or op == 0:
        return 0.0

    score = (1 + (beta ** 2)) * op / div
    return round(score, 3)
