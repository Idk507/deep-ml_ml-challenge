"""
Write a Python function that performs linear regression using gradient descent. The function should take NumPy arrays X (features with a column of ones for the intercept) and y (target) as input, along with learning rate alpha and the number of iterations, and return the coefficients of the linear regression model as a NumPy array. Round your answer to four decimal places. -0.0 is a valid result for rounding a very small number.

Example:
Input:
X = np.array([[1, 1], [1, 2], [1, 3]]), y = np.array([1, 2, 3]), alpha = 0.01, iterations = 1000
Output:
np.array([0.1107, 0.9513])
Reasoning:
The linear model is y = 0.0 + 1.0*x, which fits the input data after gradient descent optimization.

Linear Regression Using Gradient Descent
Linear regression can also be performed using a technique called gradient descent, where the coefficients (or weights) of the model are iteratively adjusted to minimize a cost function (usually mean squared error). This method is particularly useful when the number of features is too large for analytical solutions like the normal equation or when the feature matrix is not invertible.

The gradient descent algorithm updates the weights by moving in the direction of the negative gradient of the cost function with respect to the weights. The updates occur iteratively until the algorithm converges to a minimum of the cost function.

The update rule for each weight is given by:

θ
j
:
=
θ
j
−
α
1
m
∑
i
=
1
m
(
h
θ
(
x
(
i
)
)
−
y
(
i
)
)
x
j
(
i
)
θ 
j
​
 :=θ 
j
​
 −α 
m
1
​
  
i=1
∑
m
​
 (h 
θ
​
 (x 
(i)
 )−y 
(i)
 )x 
j
(i)
​
 
Explanation of Terms
( \alpha ) is the learning rate.
( m ) is the number of training examples.
( h_{\theta}(x^{(i)}) ) is the hypothesis function at iteration ( i ).
( x^{(i)} ) is the feature vector of the ( i^{\text{th}} ) training example.
( y^{(i)} ) is the actual target value for the ( i^{\text{th}} ) training example.
( x_j^{(i)} ) is the value of feature ( j ) for the ( i^{\text{th}} ) training example.
Key Points
Learning Rate: The choice of learning rate is crucial for the convergence and performance of gradient descent.
A small learning rate may lead to slow convergence.
A large learning rate may cause overshooting and divergence.
Number of Iterations: The number of iterations determines how long the algorithm runs before it converges or stops.
Practical Implementation
Implementing gradient descent involves initializing the weights, computing the gradient of the cost function, and iteratively updating the weights according to the update rule.
"""

import numpy as np
def linear_regression_gradient_descent(X: np.ndarray, y: np.ndarray, alpha: float, iterations: int) -> np.ndarray:
    m, n = X.shape
    theta = np.zeros((n, 1))
    for _ in range(iterations):
        predictions = X @ theta
        errors = predictions - y.reshape(-1, 1)
        updates = X.T @ errors / m
        theta -= alpha * updates
    return np.round(theta.flatten(), 4)
