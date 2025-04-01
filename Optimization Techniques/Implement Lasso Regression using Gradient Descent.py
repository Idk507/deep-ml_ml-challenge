"""
Implement Lasso Regression using Gradient Descent

In this problem, you need to implement the Lasso Regression algorithm using Gradient Descent. Lasso Regression (L1 Regularization) adds a penalty equal to the absolute value of the coefficients to the loss function. Your task is to update the weights and bias iteratively using the gradient of the loss function and the L1 penalty.

The objective function of Lasso Regression is:

J
(
w
,
b
)
=
1
2
n
∑
i
=
1
n
(
y
i
−
(
∑
j
=
1
p
X
i
j
w
j
+
b
)
)
2
+
α
∑
j
=
1
p
∣
w
j
∣
J(w,b)= 
2n
1
​
  
i=1
∑
n
​
 (y 
i
​
 −( 
j=1
∑
p
​
 X 
ij
​
 w 
j
​
 +b)) 
2
 +α 
j=1
∑
p
​
 ∣w 
j
​
 ∣
Where:

y
i
y 
i
​
  is the actual value for the 
i
i-th sample
y
^
i
=
∑
j
=
1
p
X
i
j
w
j
+
b
y
^
​
  
i
​
 =∑ 
j=1
p
​
 X 
ij
​
 w 
j
​
 +b is the predicted value for the 
i
i-th sample
w
j
w 
j
​
  is the weight associated with the 
j
j-th feature
α
α is the regularization parameter
b
b is the bias
Your task is to use the L1 penalty to shrink some of the feature coefficients to zero during gradient descent, thereby helping with feature selection.

Example:
Input:
import numpy as np

X = np.array([[0, 0], [1, 1], [2, 2]])
y = np.array([0, 1, 2])

alpha = 0.1
weights, bias = l1_regularization_gradient_descent(X, y, alpha=alpha, learning_rate=0.01, max_iter=1000)
Output:
(weights,bias)
(array([float, float]), float)
Reasoning:
The Lasso Regression algorithm is used to optimize the weights and bias for the given data. The weights are adjusted to minimize the loss function with the L1 penalty.

"""

"""
Understanding Lasso Regression and L1 Regularization
Lasso Regression is a type of linear regression that applies L1 regularization to the model. It adds a penalty equal to the sum of the absolute values of the coefficients, encouraging some of them to be exactly zero. This makes Lasso Regression particularly useful for feature selection, as it can shrink the coefficients of less important features to zero, effectively removing them from the model.

Steps to Implement Lasso Regression using Gradient Descent
Initialize Weights and Bias:
Start with the weights and bias set to zero.

Make Predictions:
Use the formula:

y
^
i
=
∑
j
=
1
p
X
i
j
w
j
+
b
y
^
​
  
i
​
 = 
j=1
∑
p
​
 X 
ij
​
 w 
j
​
 +b
where 
y
^
i
y
^
​
  
i
​
  is the predicted value for the 
i
i-th sample.

Compute Residuals:
Find the difference between the predicted values 
y
^
i
y
^
​
  
i
​
  and the actual values 
y
i
y 
i
​
 . These residuals are the errors in the model.

Update the Weights and Bias:
Update the weights and bias using the gradient of the loss function with respect to the weights and bias:

For weights 
w
j
w 
j
​
 :

∂
J
∂
w
j
=
1
n
∑
i
=
1
n
X
i
j
(
y
^
i
−
y
i
)
+
α
⋅
sign
(
w
j
)
∂w 
j
​
 
∂J
​
 = 
n
1
​
  
i=1
∑
n
​
 X 
ij
​
 ( 
y
^
​
  
i
​
 −y 
i
​
 )+α⋅sign(w 
j
​
 )
For bias 
b
b (without the regularization term):

∂
J
∂
b
=
1
n
∑
i
=
1
n
(
y
^
i
−
y
i
)
∂b
∂J
​
 = 
n
1
​
  
i=1
∑
n
​
 ( 
y
^
​
  
i
​
 −y 
i
​
 )
Update the weights and bias:

w
j
=
w
j
−
η
⋅
∂
J
∂
w
j
w 
j
​
 =w 
j
​
 −η⋅ 
∂w 
j
​
 
∂J
​
 
b
=
b
−
η
⋅
∂
J
∂
b
b=b−η⋅ 
∂b
∂J
​
 
Check for Convergence:
The algorithm stops when the L1 norm of the gradient with respect to the weights becomes smaller than a predefined threshold 
tol
tol:

∣
∣
∇
w
∣
∣
1
=
∑
j
=
1
p
∣
∂
J
∂
w
j
∣
∣∣∇w∣∣ 
1
​
 = 
j=1
∑
p
​
  
​
  
∂w 
j
​
 
∂J
​
  
​
 
Return the Weights and Bias:
Once the algorithm converges, return the optimized weights and bias.

"""

import numpy as np

def l1_regularization_gradient_descent(X: np.array, y: np.array, alpha: float = 0.1, learning_rate: float = 0.01, max_iter: int = 1000, tol: float = 1e-4) -> tuple:
	n_samples, n_features = X.shape

	weights = np.zeros(n_features)
	bias = 0
	for iterations in range(max_iter):
        y_pred = np.dot(X,weights)+bias 
        error = y_pred - y 
        grad_w = (1/n_samples)* np.dot(X.T,error)+alpha* np.sign(weights)
        grad_b = (1/n_samples)* np.sum(error)
        weights -= learning_rate*grad_w 
        bias -= learning_rate*grad_b 

        if np.linalg.norm(grad_w,ord=1) < tol:
            break 
    return weights,bias
