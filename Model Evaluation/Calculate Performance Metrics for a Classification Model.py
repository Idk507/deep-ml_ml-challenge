"""
Calculate Performance Metrics for a Classification Model

Task: Implement Performance Metrics Calculation
In this task, you are required to implement a function performance_metrics(actual, predicted) that computes various performance metrics for a binary classification problem. These metrics include:

Confusion Matrix
Accuracy
F1 Score
Specificity
Negative Predictive Value
The function should take in two lists:

actual: The actual class labels (1 for positive, 0 for negative).
predicted: The predicted class labels from the model.
Output
The function should return a tuple containing:

confusion_matrix: A 2x2 matrix.
accuracy: A float representing the accuracy of the model.
f1_score: A float representing the F1 score of the model.
specificity: A float representing the specificity of the model.
negative_predictive_value: A float representing the negative predictive value.
Constraints
All elements in the actual and predicted lists must be either 0 or 1.
Both lists must have the same length.
Example:
Input:
actual = [1, 0, 1, 0, 1]
predicted = [1, 0, 0, 1, 1]
print(performance_metrics(actual, predicted))
Output:
([[2, 1], [1, 1]], 0.6, 0.667, 0.5, 0.5)

"""
"""
Performance Metrics
Performance metrics such as accuracy, F1 score, specificity, negative predictive value, precision, and recall are vital to understanding how a model is performing.

How many observations are correctly labeled? Are we mislabeling one category more than the other? Performance metrics can answer these questions and provide an idea of where to focus to improve a model's performance.

For this problem, starting with the confusion matrix is a helpful first step, as all the elements of the confusion matrix can help with calculating other performance metrics.

For a binary classification problem of a dataset with 
n
n observations, the confusion matrix is a 
2
×
2
2×2 matrix with the following structure:

M
=
(
T
P
F
N
F
P
T
N
)
M=( 
TP
FP
​
  
FN
TN
​
 )
Where:

TP: True positives, the number of observations from the positive label that were correctly labeled as positive.
FN: False negatives, the number of observations from the positive label that were incorrectly labeled as negative.
FP: False positives, the number of observations from the negative label that were incorrectly labeled as positive.
TN: True negatives, the number of observations from the negative label that were correctly labeled as negative.
Metrics
Accuracy
How many observations are labeled as the actual category they belong to?

Accuracy
=
T
P
+
T
N
T
P
+
T
N
+
F
P
+
F
N
Accuracy= 
TP+TN+FP+FN
TP+TN
​
 
Precision
How many elements labeled as positive are actually positive?

Precision
=
T
P
T
P
+
F
P
Precision= 
TP+FP
TP
​
 
Negative Predictive Value
How many elements labeled as negative are actually negative?

Negative Predictive Value
=
T
N
T
N
+
F
N
Negative Predictive Value= 
TN+FN
TN
​
 
Recall
Out of all positive elements, how many were correctly labeled?

Recall
=
T
P
T
P
+
F
N
Recall= 
TP+FN
TP
​
 
Specificity
How well are we labeling the negative elements correctly?

Specificity
=
T
N
T
N
+
F
P
Specificity= 
TN+FP
TN
​
 
F1 Score
How to account for the trade-off of false negatives and positives? The F1 score is the harmonic mean of precision and recall.

F1 Score
=
2
×
Precision
×
Recall
Precision
+
Recall
F1 Score=2× 
Precision+Recall
Precision×Recall
​


"""


from collections import Counter

def performance_metrics(actual: list[int], predicted: list[int]) -> tuple:
    data = list(zip(actual, predicted))
    counts = Counter(tuple(pair) for pair in data)
    TP, FN, FP, TN = counts[(1, 1)], counts[(1, 0)], counts[(0, 1)], counts[(0, 0)]
    confusion_matrix = [[TP, FN], [FP, TN]]
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * precision * recall / (precision + recall)
    negativePredictive = TN / (TN + FN)
    specificity = TN / (TN + FP)
    return confusion_matrix, round(accuracy, 3), round(f1, 3), round(specificity, 3), round(negativePredictive, 3)
