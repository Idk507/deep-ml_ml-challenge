"""
Pegasos Kernel SVM Implementation

Write a Python function that implements the Pegasos algorithm to train a kernel SVM classifier from scratch. The function should take a dataset (as a 2D NumPy array where each row represents a data sample and each column represents a feature), a label vector (1D NumPy array where each entry corresponds to the label of the sample), and training parameters such as the choice of kernel (linear or RBF), regularization parameter (lambda), and the number of iterations. The function should perform binary classification and return the model's alpha coefficients and bias.

Example:
Input:
data = np.array([[1, 2], [2, 3], [3, 1], [4, 1]]), labels = np.array([1, 1, -1, -1]), kernel = 'rbf', lambda_val = 0.01, iterations = 100
Output:
alpha = [0.03, 0.02, 0.05, 0.01], b = -0.05
Reasoning:
Using the RBF kernel, the Pegasos algorithm iteratively updates the weights based on a sub-gradient descent method, taking into account the non-linear separability of the data induced by the kernel transformation.
"""
"""
Pegasos Algorithm and Kernel SVM
The Pegasos (Primal Estimated sub-GrAdient SOlver for SVM) algorithm is a simple and efficient stochastic gradient descent method designed for solving the SVM optimization problem in its primal form.

Key Concepts
Kernel Trick: Allows SVM to classify data that is not linearly separable by implicitly mapping input features into high-dimensional feature spaces.
Regularization Parameter ( \lambda ): Controls the trade-off between achieving a low training error and a low model complexity.
Sub-Gradient Descent: Used in the Pegasos algorithm to optimize the objective function, which includes both the hinge loss and a regularization term.
Steps in the Pegasos Algorithm
Initialization: Set weights to zero and choose an appropriate value for the regularization parameter ( \lambda ).
Iterative Updates: For each iteration and for each randomly selected example:
Perform updates using the learning rule derived from the sub-gradient of the loss function.
Kernel Computation: Use the chosen kernel to compute the dot products required in the update step, enabling non-linear decision boundaries.
Practical Implementation
The implementation involves:

Selecting a kernel function.
Calculating the kernel matrix.
Performing iterative updates on the alpha coefficients using the Pegasos rule:
α
t
+
1
=
(
1
−
η
t
λ
)
α
t
+
η
t
(
y
i
K
(
x
i
,
x
)
)
α 
t+1
​
 =(1−η 
t
​
 λ)α 
t
​
 +η 
t
​
 (y 
i
​
 K(x 
i
​
 ,x))
where ( \eta_t ) is the learning rate at iteration ( t ), and ( K ) denotes the kernel function.
Advantages
This method is particularly well-suited for large-scale learning problems due to its efficient use of data and incremental learning nature.

"""

import numpy as np

def linear_kernel(x, y):
    return np.dot(x, y)

def rbf_kernel(x, y, sigma=1.0):
    return np.exp(-np.linalg.norm(x-y)**2 / (2 * (sigma ** 2)))

def pegasos_kernel_svm(data, labels, kernel='linear', lambda_val=0.01, iterations=100, sigma=1.0):
    n_samples = len(data)
    alphas = np.zeros(n_samples)
    b = 0

    for t in range(1, iterations + 1):
        for i in range(n_samples):
            eta = 1.0 / (lambda_val * t)
            if kernel == 'linear':
                kernel_func = linear_kernel
            elif kernel == 'rbf':
                kernel_func = lambda x, y: rbf_kernel(x, y, sigma)
    
            decision = sum(alphas[j] * labels[j] * kernel_func(data[j], data[i]) for j in range(n_samples)) + b
            if labels[i] * decision < 1:
                alphas[i] += eta * (labels[i] - lambda_val * alphas[i])
                b += eta * labels[i]

    return np.round(alphas,4).tolist(), np.round(b,4)
