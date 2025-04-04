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
Reasoning:
The function calculates the confusion matrix, accuracy, F1 score, specificity, and negative predictive value based on the input labels. The resulting values are rounded to three decimal places as required.


"""

class Solution:
    def minFlips(self, a: int, b: int, c: int) -> int:
        max_bit = max(a.bit_length(),b.bit_length(),c.bit_length())
        r = 0
        
        for i in range(max_bit):
            bit_a = (a >> i)& 1
            bit_b = (b>> i)& 1
            bit_c = (c >> i) &1 

            if (bit_a | bit_b) != bit_c :
                if bit_c == 1:
                    r +=1 
                else : 
                    r += bit_a + bit_b 
        return r
