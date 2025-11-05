"""
Write a Python function that implements a deterministic version of the Pegasos algorithm to train a kernel SVM classifier from scratch. The function should take a dataset (as a 2D NumPy array where each row represents a data sample and each column represents a feature), a label vector (1D NumPy array where each entry corresponds to the label of the sample), and training parameters such as the choice of kernel (linear or RBF), regularization parameter (lambda), and the number of iterations. Note that while the original Pegasos algorithm is stochastic (it selects a single random sample at each step), this problem requires using all samples in every iteration (i.e., no random sampling). The function should perform binary classification and return the model's alpha coefficients and bias.

Example:
Input:
data = np.array([[1, 2], [2, 3], [3, 1], [4, 1]]), labels = np.array([1, 1, -1, -1]), kernel = 'rbf', lambda_val = 0.01, iterations = 100, sigma = 1.0
Output:
alpha = [0.03, 0.02, 0.05, 0.01], b = -0.05
Reasoning:
Using the RBF kernel, the Pegasos algorithm iteratively updates the weights based on a sub-gradient descent method, taking into account the non-linear separability of the data induced by the kernel transformation.

Pegasos Algorithm for Kernel SVM (Deterministic Version)
Introduction
The Pegasos Algorithm (Primal Estimated sub-GrAdient SOlver for SVM) is a fast, iterative algorithm designed to train Support Vector Machines (SVM). While the original Pegasos algorithm uses stochastic updates by selecting one random sample per iteration, this problem requires a deterministic versionmeaning every data sample is evaluated and considered in each iteration. This deterministic approach ensures reproducibility and clarity, particularly useful for educational purposes.

Key Concepts
Kernel Trick:
SVM typically separates data classes using a linear hyperplane. However, real-world data isn't always linearly separable. The Kernel Trick implicitly maps input data into a higher-dimensional feature space, making it easier to separate non-linear data.

Common kernel functions include:

Linear Kernel: 
K
(
x
,
y
)
=
x
⋅
y
K(x,y)=x⋅y
Radial Basis Function (RBF) Kernel: 
K
(
x
,
y
)
=
e
−
∥
x
−
y
∥
2
2
σ
2
K(x,y)=e 
− 
2σ 
2
 
∥x−y∥ 
2
 
​
 
 
Regularization Parameter (
λ
λ):
This parameter balances how closely the model fits training data against the complexity of the model, helping to prevent overfitting.

Sub-gradient Descent:
Pegasos optimizes the SVM objective function using iterative parameter updates based on the sub-gradient of the hinge loss.

Deterministic Pegasos Algorithm Steps
Given training samples 
(
x
i
,
y
i
)
(x 
i
​
 ,y 
i
​
 ), labels 
y
i
∈
{
−
1
,
1
}
y 
i
​
 ∈{−1,1}, kernel function 
K
K, regularization parameter 
λ
λ, and total iterations 
T
T:

Initialize alpha coefficients 
α
i
=
0
α 
i
​
 =0 and bias 
b
=
0
b=0.
For each iteration 
t
=
1
,
2
,
…
,
T
t=1,2,…,T:
Compute learning rate: 
η
t
=
1
λ
t
η 
t
​
 = 
λt
1
​
 
For each training sample 
(
x
i
,
y
i
)
(x 
i
​
 ,y 
i
​
 ):
Compute decision value: 
f
(
x
i
)
=
∑
j
α
j
y
j
K
(
x
j
,
x
i
)
+
b
f(x 
i
​
 )=∑ 
j
​
 α 
j
​
 y 
j
​
 K(x 
j
​
 ,x 
i
​
 )+b
If the margin constraint 
y
i
f
(
x
i
)
<
1
y 
i
​
 f(x 
i
​
 )<1 is violated, update parameters:
α
i
←
α
i
+
η
t
(
y
i
−
λ
α
i
)
α 
i
​
 ←α 
i
​
 +η 
t
​
 (y 
i
​
 −λα 
i
​
 )
b
←
b
+
η
t
y
i
b←b+η 
t
​
 y 
i
​
 
Example (Conceptual Explanation)
Consider a simple dataset:

Data:
X
=
[
[
1
,
2
]
,
[
2
,
3
]
,
[
3
,
1
]
,
[
4
,
1
]
]
X=[[1,2],[2,3],[3,1],[4,1]], 
Y
=
[
1
,
1
,
−
1
,
−
1
]
Y=[1,1,−1,−1]

Parameters: Linear kernel, 
λ
=
0.01
λ=0.01, iterations = 
1
1

Initially, parameters (
α
,
b
α,b) start at zero. For each sample, you calculate the decision value. Whenever a sample violates the margin constraint (
y
i
f
(
x
i
)
<
1
y 
i
​
 f(x 
i
​
 )<1), you update the corresponding 
α
i
α 
i
​
  and bias 
b
b as described. After looping through all samples for the specified iterations, you obtain the trained parameters.

Important Implementation Notes:
Always iterate through all samples in every iteration (no stochastic/random sampling).
Clearly distinguish kernel function choices in your implementation.
After training, predictions for new data 
x
x are made using:
y
^
(
x
)
=
sign
(
∑
j
α
j
y
j
K
(
x
j
,
x
)
+
b
)
y
^
​
 (x)=sign( 
j
∑
​
 α 
j
​
 y 
j
​
 K(x 
j
​
 ,x)+b)
This deterministic Pegasos variant clearly demonstrates how kernelized SVM training operates and simplifies the understanding of kernel methods.

"""
import numpy as np

def pegasos_kernel_svm(data: np.ndarray, labels: np.ndarray, kernel='linear', lambda_val=0.01, iterations=100, sigma=1.0) -> (list, float):
    n_samples = data.shape[0]
    alphas = np.zeros(n_samples)
    b = 0.0

    # Define kernel functions
    def linear_kernel(x, y):
        return np.dot(x, y)

    def rbf_kernel(x, y):
        return np.exp(-np.linalg.norm(x - y) ** 2 / (2 * sigma ** 2))

    # Precompute the kernel matrix
    K = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(n_samples):
            if kernel == 'linear':
                K[i, j] = linear_kernel(data[i], data[j])
            elif kernel == 'rbf':
                K[i, j] = rbf_kernel(data[i], data[j])
            else:
                raise ValueError("Unsupported kernel type. Choose 'linear' or 'rbf'.")

    # Training loop
    for t in range(1, iterations + 1):
        eta_t = 1.0 / (lambda_val * t)
        for i in range(n_samples):
            decision_value = np.sum(alphas * labels * K[:, i]) + b
            if labels[i] * decision_value < 1:
                alphas[i] += eta_t * (labels[i] - lambda_val * alphas[i])
                b += eta_t * labels[i]

    return alphas.tolist(), b
