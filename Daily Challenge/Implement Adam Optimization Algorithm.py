"""
Implement Adam Optimization Algorithm

Implement the Adam (Adaptive Moment Estimation) optimization algorithm in Python. Adam is an optimization algorithm that adapts the learning rate for each parameter. Your task is to write a function adam_optimizer that updates the parameters of a given function using the Adam algorithm.

The function should take the following parameters:

f: The objective function to be optimized
grad: A function that computes the gradient of f
x0: Initial parameter values
learning_rate: The step size (default: 0.001)
beta1: Exponential decay rate for the first moment estimates (default: 0.9)
beta2: Exponential decay rate for the second moment estimates (default: 0.999)
epsilon: A small constant for numerical stability (default: 1e-8)
num_iterations: Number of iterations to run the optimizer (default: 1000)
The function should return the optimized parameters.

Example:
Input:
import numpy as np

def objective_function(x):
    return x[0]**2 + x[1]**2

def gradient(x):
    return np.array([2*x[0], 2*x[1]])

x0 = np.array([1.0, 1.0])
x_opt = adam_optimizer(objective_function, gradient, x0)

print("Optimized parameters:", x_opt)
Output:
# Optimized parameters: [0.99000325 0.99000325]
Reasoning:
The Adam optimizer updates the parameters to minimize the objective function. In this case, the objective function is the sum of squares of the parameters, and the optimizer finds the optimal values for the parameters

"""

"""
Understanding the Adam Optimization Algorithm
Adam (Adaptive Moment Estimation) is an optimization algorithm commonly used in training deep neural networks. It combines ideas from two other optimization algorithms: RMSprop and Momentum.

Key Concepts
Adaptive Learning Rates:
Adam computes individual adaptive learning rates for different parameters from estimates of first and second moments of the gradients.
Momentum:
It keeps track of an exponentially decaying average of past gradients, similar to momentum.
RMSprop:
It also keeps track of an exponentially decaying average of past squared gradients.
Bias Correction:
Adam includes bias correction terms to account for the initialization of the first and second moment estimates.
The Adam Algorithm
Given parameters 
θ
θ, objective function 
f
(
θ
)
f(θ), and its gradient 
∇
θ
f
(
θ
)
∇ 
θ
​
 f(θ):

Initialize:
Time step 
t
=
0
t=0
Parameters 
θ
0
θ 
0
​
 
First moment vector 
m
0
=
0
m 
0
​
 =0
Second moment vector 
v
0
=
0
v 
0
​
 =0
Hyperparameters 
α
α (learning rate), 
β
1
β 
1
​
 , 
β
2
β 
2
​
 , and 
ϵ
ϵ
While not converged, do:
Increment time step: 
t
=
t
+
1
t=t+1
Compute gradient: 
g
t
=
∇
θ
f
t
(
θ
t
−
1
)
g 
t
​
 =∇ 
θ
​
 f 
t
​
 (θ 
t−1
​
 )
Update biased first moment estimate: 
m
t
=
β
1
⋅
m
t
−
1
+
(
1
−
β
1
)
⋅
g
t
m 
t
​
 =β 
1
​
 ⋅m 
t−1
​
 +(1−β 
1
​
 )⋅g 
t
​
 
Update biased second raw moment estimate: 
v
t
=
β
2
⋅
v
t
−
1
+
(
1
−
β
2
)
⋅
g
t
2
v 
t
​
 =β 
2
​
 ⋅v 
t−1
​
 +(1−β 
2
​
 )⋅g 
t
2
​
 
Compute bias-corrected first moment estimate: 
m
^
t
=
m
t
/
(
1
−
β
1
t
)
m
^
  
t
​
 =m 
t
​
 /(1−β 
1
t
​
 )
Compute bias-corrected second raw moment estimate: 
v
^
t
=
v
t
/
(
1
−
β
2
t
)
v
^
  
t
​
 =v 
t
​
 /(1−β 
2
t
​
 )
Update parameters: 
θ
t
=
θ
t
−
1
−
α
⋅
m
^
t
/
(
v
^
t
+
ϵ
)
θ 
t
​
 =θ 
t−1
​
 −α⋅ 
m
^
  
t
​
 /( 
v
^
  
t
​
 
​
 +ϵ)
Adam combines the advantages of AdaGrad, which works well with sparse gradients, and RMSProp, which works well in online and non-stationary settings. Adam is generally regarded as being fairly robust to the choice of hyperparameters, though the learning rate may sometimes need to be changed from the suggested default.
"""
import numpy as np

def objective_function(x):
    return x[0]**2 + x[1]**2

def gradient(x):
    return np.array([2*x[0], 2*x[1]])

x0 = np.array([1.0, 1.0])
x_opt = adam_optimizer(objective_function, gradient, x0)

print("Optimized parameters:", x_opt)

import numpy as np

def adam_optimizer(f, grad, x0, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, num_iterations=10):
	# Your code here
	x = x0
    m = np.zeros_like(x)
    n = np.zeros_like(x)
    for t in range(1,num_iterations+1):
        g = grad(x)
        m = beta1*m + (1 - beta1)*g 
        n = beta2*n + (1-beta2)*g**2
        m_hat = m / (1 - beta1 ** t)
        n_hat = n / (1 - beta2**t)
        x = x - learning_rate * m_hat /(np.sqrt(n_hat)+epsilon)
    return x


