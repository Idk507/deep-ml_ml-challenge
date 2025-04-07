"""
Binary Classification with Logistic Regression

Implement the prediction function for binary classification using Logistic Regression. Your task is to compute class probabilities using the sigmoid function and return binary predictions based on a threshold of 0.5.

Example:
Input:
predict_logistic(np.array([[1, 1], [2, 2], [-1, -1], [-2, -2]]), np.array([1, 1]), 0)
Output:
[1 1 0 0]
Reasoning:
Each sample's linear combination is computed using 
z
=
X
w
+
b
z=Xw+b. The sigmoid function is applied, and the output is thresholded at 0.5, resulting in binary predictions

"""
"""
Binary Classification with Logistic Regression
Logistic Regression is a fundamental algorithm for binary classification. Given input features and learned model parameters (weights and bias), your task is to implement the prediction function that computes class probabilities.

Mathematical Background
The logistic regression model makes predictions using the sigmoid function:

σ
(
z
)
=
1
1
+
e
−
z
σ(z)= 
1+e 
−z
 
1
​
 

where z is the linear combination of features and weights plus bias:

z
=
w
T
x
+
b
=
∑
i
=
1
n
w
i
x
i
+
b
z=w 
T
 x+b=∑ 
i=1
n
​
 w 
i
​
 x 
i
​
 +b

Implementation Requirements
Your task is to implement a function that:

Takes a batch of samples 
X
X (shape: N x D), weights 
w
w (shape: D), and bias b
Computes 
z
=
X
w
+
b
z=Xw+b for all samples
Applies the sigmoid function to get probabilities
Returns binary predictions i.e., 0 or 1 using a threshold of 0.5
Important Considerations
Handle numerical stability in sigmoid computation
Ensure efficient vectorized operations using numpy
Return binary predictions (0 or 1)
Hint
To prevent overflow in the exponential calculation of the sigmoid function, use np.clip to limit z values:

z = np.clip(z, -500, 500)
This ensures numerical stability when dealing with large input values.
"""
import numpy as np

def predict_logistic(X: np.ndarray, weights: np.ndarray, bias: float) -> np.ndarray:
    """
    Implements binary classification prediction using Logistic Regression.

    Args:
        X: Input feature matrix (shape: N Ã D)
        weights: Model weights (shape: D)
        bias: Model bias

    Returns:
        Binary predictions (0 or 1)
    """
    z = np.dot(X, weights) + bias
    z = np.clip(z, -500, 500)  # Prevent overflow in exp
    probabilities = 1 / (1 + np.exp(-z))
    return (probabilities >= 0.5).astype(int)
