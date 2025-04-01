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
def dot(v1, v2):
    return sum([ax1 * ax2 for ax1, ax2 in zip(v1, v2)])

def scalar_mult(scalar, v):
    return [scalar * ax for ax in v]

def orthogonal_projection(v, L):
    L_mag_sq = dot(L, L)
    proj_scalar = dot(v, L) / L_mag_sq
    proj_v = scalar_mult(proj_scalar, L)
    return [round(x, 3) for x in proj_v]
