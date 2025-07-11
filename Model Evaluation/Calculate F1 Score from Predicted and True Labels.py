"""
Calculate F1 Score from Predicted and True Labels
Implement a function to calculate the F1 score given predicted and true labels. The F1 score is a widely used metric in machine learning, combining precision and recall into a single measure. round your solution to the 3rd decimal place

Example:
Input:
y_true = [1, 0, 1, 1, 0], y_pred = [1, 0, 0, 1, 1]
Output:
0.667
Reasoning:
The true positives, false positives, and false negatives are calculated from the given labels. Precision and recall are derived, and the F1 score is computed as their harmonic mean.
F1 Score
The F1 score is a widely used metric in machine learning and statistics, particularly for evaluating classification models. It is the harmonic mean of precision and recall, providing a single measure that balances the trade-off between these two metrics.

Key Concepts
Precision: Precision is the fraction of true positive predictions out of all positive predictions made by the model. It measures how many of the predicted positive instances are actually correct.

Precision
=
True Positives (TP)
True Positives (TP)
+
False Positives (FP)
Precision= 
True Positives (TP)+False Positives (FP)
True Positives (TP)
​
 
Recall: Recall is the fraction of true positive predictions out of all actual positive instances in the dataset. It measures how many of the actual positive instances were correctly predicted.

Recall
=
True Positives (TP)
True Positives (TP)
+
False Negatives (FN)
Recall= 
True Positives (TP)+False Negatives (FN)
True Positives (TP)
​
 
F1 Score: The F1 score is the harmonic mean of precision and recall, providing a balanced measure that takes both metrics into account:

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
 
Why Use the F1 Score?
The F1 score is particularly useful when the dataset is imbalanced, meaning the classes are not equally represented. It provides a single metric that balances the trade-off between precision and recall, especially in scenarios where maximizing one metric might lead to a significant drop in the other.

Example Calculation
Given: y_true = [1, 0, 1, 1, 0]

y_pred = [1, 0, 0, 1, 1]

Calculate True Positives (TP), False Positives (FP), and False Negatives (FN):

TP
=
2
,
FP
=
1
,
FN
=
1
TP=2,FP=1,FN=1
Calculate Precision:

Precision
=
2
2
+
1
=
2
3
≈
0.667
Precision= 
2+1
2
​
 = 
3
2
​
 ≈0.667
Calculate Recall:

Recall
=
2
2
+
1
=
2
3
≈
0.667
Recall= 
2+1
2
​
 = 
3
2
​
 ≈0.667
Calculate F1 Score:

F1 Score
=
2
×
0.667
×
0.667
0.667
+
0.667
=
0.667
F1 Score=2× 
0.667+0.667
0.667×0.667
​
 =0.667
Applications
The F1 score is widely used in:

Binary classification problems (e.g., spam detection, fraud detection).
Multi-class classification problems (evaluated per class and averaged).
Information retrieval tasks (e.g., search engines, recommendation systems).
Mastering the F1 score is essential for evaluating and comparing the performance of classification models.

"""
def calculate_f1_score(y_true, y_pred):
	"""
	Calculate the F1 score based on true and predicted labels.

	Args:
		y_true (list): True labels (ground truth).
		y_pred (list): Predicted labels.

	Returns:
		float: The F1 score rounded to three decimal places.
	"""
	# Your code here
	tp = sum((yt == 1 and yp == 1)for yt,yp in zip(y_true,y_pred))
    fp = sum((yt == 0 and yp == 1)for yt,yp in zip(y_true,y_pred))
    fn = sum((yt == 1 and yp == 0)for yt,yp in zip(y_true,y_pred))
    pre = tp/(tp+fp) if (tp+fp) != 0 else 0
    rec = tp / (tp+fn) if (tp+fn) != 0 else 0
    f1 = 2*(pre*rec) / (pre+rec) if (pre+rec) != 0 else 0.0
    return round(f1,3)
