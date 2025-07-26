"""
The Muon optimizer is an algorithm that combines momentum with a matrix preconditioning step based on the Newton-Schulz iteration. In this task, you will implement a single Muon optimizer update for a 2D NumPy array of parameters. Your implementation should:

Update the momentum using the current gradient and previous momentum.
Apply the Newton-Schulz matrix iteration (order 5) to precondition the update direction. This involves normalizing the update, possibly transposing for wide matrices, and running a fixed matrix iteration for a number of steps.
Use a scale factor based on the RMS operator norm for stability.
Update the parameters using the preconditioned direction, learning rate, and scale.
Return both the updated parameter matrix and momentum.

Example:
Input:
theta = np.eye(2)
B = np.zeros((2,2))
grad = np.ones((2,2))
eta = 0.1
mu = 0.9
theta_new, B_new = muon_step(theta, B, grad, eta, mu, ns_steps=2)
print(np.round(theta_new, 3))
Output:
[[ 0.944 -0.056] [-0.056 0.944]]
Reasoning:
After the momentum and Newton-Schulz preconditioning, the parameters are updated in the direction of the scaled matrix.

Muon Optimizer: Mathematical Foundations
1. Introduction
The Muon optimizer is a gradient-based optimization algorithm that combines momentum with a matrix preconditioning step based on the Newton-Schulz iteration. The goal is to stabilize and speed up neural network training, especially when parameters are matrices (such as in fully connected and convolutional layers).

2. Step 1: Momentum Update
Muon uses the momentum method to smooth the stochasticity of gradients:

M
t
=
β
⋅
M
t
−
1
+
(
1
−
β
)
⋅
g
t
M 
t
​
 =β⋅M 
t−1
​
 +(1−β)⋅g 
t
​
 
M
t
M 
t
​
 : Momentum at step 
t
t
g
t
g 
t
​
 : Current gradient
β
β: Momentum decay coefficient (e.g., 0.9â0.99)
With Nesterov momentum, the update direction is:

U
t
=
(
1
−
β
)
⋅
g
t
+
β
⋅
M
t
U 
t
​
 =(1−β)⋅g 
t
​
 +β⋅M 
t
​
 
3. Step 2: Newton-Schulz Matrix Preconditioning
For matrix-shaped gradients (or reshaped higher-dimensional gradients), Muon applies a Newton-Schulz iteration. This is an iterative algorithm that, in this context, "orthogonalizes" or stabilizes the update direction without needing explicit inversion or SVD.

A. Frobenius Normalization
First, the update matrix 
X
X is normalized by its Frobenius norm to avoid scale explosion:

X
0
=
U
t
∥
U
t
∥
F
+
ε
X 
0
​
 = 
∥U 
t
​
 ∥ 
F
​
 +ε
U 
t
​
 
​
 
where 
∥
U
t
∥
F
=
∑
i
,
j
(
U
t
)
i
j
2
∥U 
t
​
 ∥ 
F
​
 = 
∑ 
i,j
​
 (U 
t
​
 ) 
ij
2
​
 
​
 , and 
ε
ε is a small constant for stability.

B. Quintic Newton-Schulz Iteration
The following update is performed for 
k
=
1
k=1 to 
N
N steps (often 
N
=
5
N=5):

A
=
X
k
X
k
⊤
B
=
b
⋅
A
+
c
⋅
(
A
A
)
X
k
+
1
=
a
⋅
X
k
+
B
X
k
A=X 
k
​
 X 
k
⊤
​
 
B=b⋅A+c⋅(AA)
X 
k+1
​
 =a⋅X 
k
​
 +BX 
k
​
 
with fixed coefficients:

a
=
3.4445
a=3.4445
b
=
−
4.7750
b=−4.7750
c
=
2.0315
c=2.0315
This process "pushes" 
X
X closer to an orthogonal-like matrix, improving the update's conditioning.

C. Optional Reshape
If the parameter tensor is 4D (common in conv layers), it is reshaped into a 2D matrix for preconditioning, then reshaped back after.

4. Step 3: Parameter Update
Finally, the parameter update is:

θ
t
=
θ
t
−
1
−
η
⋅
S
t
θ 
t
​
 =θ 
t−1
​
 −η⋅S 
t
​
 
where 
S
t
S 
t
​
  is the matrix after Newton-Schulz preconditioning and 
η
η is the learning rate.

5. Summary
Momentum: Smooths the update direction by combining past and current gradients.
Matrix Preconditioning: Applies an iterative matrix operation to stabilize and orthogonalize the update, reducing problems due to ill-conditioning.
Parameter Update: Uses the preconditioned matrix to update model parameters.
Muon is especially effective for matrix-shaped weights and large neural networks, improving stability and potentially accelerating convergence.
"""
import numpy as np
def newton_schulz5(G, steps=5, eps=1e-7):
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.astype(np.float32)
    X /= np.linalg.norm(X, 'fro') + eps
    transposed = False
    if X.shape[0] > X.shape[1]:
        X = X.T
        transposed = True
    for _ in range(steps):
        A = X @ X.T
        X = a * X + (b * A + c * A @ A) @ X
    if transposed:
        X = X.T
    return X

def muon_step(theta, B, grad, eta, mu, ns_steps=5, eps=1e-7):
    B_new = mu * B + grad
    O = newton_schulz5(B_new, steps=ns_steps)
    scale = np.sqrt(np.prod(theta.shape)) / (np.linalg.norm(B_new, 'fro') + eps)
    theta_new = theta - eta * scale * O
    return theta_new, B_new
