"""
Train Softmax Regression with Gradient Descent
Implement a gradient descent-based training algorithm for Softmax regression. Your task is to compute model parameters using Cross Entropy loss and return the optimized coefficients along with collected loss values over iterations. Make sure to round your solution to 4 decimal places

Example:
Input:
train_softmaxreg(np.array([[0.5, -1.2], [-0.3, 1.1], [0.8, -0.6]]), np.array([0, 1, 2]), 0.01, 10)
Output:
([[-0.0011, 0.0145, -0.0921], [0.002, -0.0598, 0.1263], [-0.0009, 0.0453, -0.0342]], [3.2958, 3.2611, 3.2272, 3.1941, 3.1618, 3.1302, 3.0993, 3.0692, 3.0398, 3.011])
Reasoning:
The function iteratively updates the Softmax regression parameters using gradient descent and collects loss values over iterations.
Overview
Softmax regression is a type of logistic regression that extends it to a multiclass problem by outputting a vector 
P
P of probabilities for each distinct class and taking 
a
r
g
m
a
x
(
P
)
argmax(P).

Connection to a regular logistic regression
Recall that a standard logistic regression is aimed at approximating

p
=
1
e
−
X
β
+
1
=
=
e
X
β
1
+
e
X
β
,
p= 
e 
−Xβ
 +1
1
​
 =
= 
1+e 
Xβ
 
e 
Xβ
 
​
 ,
which actually alignes with the definition of the softmax function:

s
o
f
t
m
a
x
(
z
i
)
=
σ
(
z
i
)
=
e
z
i
∑
j
C
e
z
j
,
softmax(z 
i
​
 )=σ(z 
i
​
 )= 
∑ 
j
C
​
 e 
z 
j
​
 
 
e 
z 
i
​
 
 
​
 ,
where 
C
C is the number of classes and values of which sum up to 
1
1. Hence it simply extends the functionality of sigmoid to more than 2 classes and could be used for assigning probability values in a categorical distribution, i.e. softmax regression searches for the following vector-approximation:

p
(
i
)
=
e
x
(
i
)
β
∑
j
C
e
j
x
(
i
)
β
j
p 
(i)
 = 
∑ 
j
C
​
 e 
j
x 
(i)
 β 
j
​
 
​
 
e 
x 
(i)
 β
 
​
 
Loss in softmax regression
tl;dr key differences in the loss from logistic regression include replacing sigmoid with softmax and calculating several gradients for vectors 
β
j
β 
j
​
  corresponding to a particular class 
j
∈
{
1
,
.
.
.
,
C
}
j∈{1,...,C}.

Recall that we use MLE in logistic regression. It is the same case with softmax regression, although instead of Bernoulli-distributed random variable we have categorical distribution, which is an extension of Bernoulli to more than 2 labels. Its PMF is defined as:

f
(
y
∣
p
)
=
∏
i
=
1
K
p
i
[
i
=
y
]
,
f(y∣p)= 
i=1
∏
K
​
 p 
i
[i=y]
​
 ,
Hence, our log-likelihood looks like:

∑
X
∑
j
C
[
y
i
=
j
]
log
⁡
[
p
(
x
i
)
]
X
∑
​
  
j
∑
C
​
 [y 
i
​
 =j]log[p(x 
i
​
 )]
Where we replace our probability function with softmax:

∑
X
∑
j
C
[
y
i
=
j
]
log
⁡
e
x
i
β
j
∑
j
C
e
x
i
β
j
X
∑
​
  
j
∑
C
​
 [y 
i
​
 =j]log 
∑ 
j
C
​
 e 
x 
i
​
 β 
j
​
 
 
e 
x 
i
​
 β 
j
​
 
 
​
 
where 
[
i
=
y
]
[i=y] is a function, that returns 
0
0, if 
i
≠
y
i

=y, and 
1
1 otherwise and 
C
C - number of distinct classes (labels). You can see that since we are expecting a 
1
×
C
1×C output of 
y
y, just like in the neuron backprop problem, we will be having separate vector 
β
j
β 
j
​
  for every 
j
j class out of 
C
C.

Optimization objective
The optimization objective is the same as with logistic regression. The function, which we are optimizing, is also commonly refered as Cross Entropy (CE):

a
r
g
m
i
n
β
−
[
∑
X
∑
j
C
[
y
i
=
j
]
log
⁡
e
x
i
β
j
∑
j
C
e
x
i
β
j
]
argmin 
β
​
 −[ 
X
∑
​
  
j
∑
C
​
 [y 
i
​
 =j]log 
∑ 
j
C
​
 e 
x 
i
​
 β 
j
​
 
 
e 
x 
i
​
 β 
j
​
 
 
​
 ]
Then we are yet again using a chain rule for calculating partial derivative of 
C
E
CE with respect to 
β
β:

∂
C
E
∂
β
i
(
j
)
=
∂
C
E
∂
σ
∂
σ
∂
[
X
β
(
j
)
]
∂
[
X
β
(
j
)
]
β
i
(
j
)
∂β 
i
(j)
​
 
∂CE
​
 = 
∂σ
∂CE
​
  
∂[Xβ 
(j)
 ]
∂σ
​
  
β 
i
(j)
​
 
∂[Xβ 
(j)
 ]
​
 
Which is eventually reduced to a similar to logistic regression gradient matrix form:

X
T
(
σ
(
X
β
(
j
)
)
−
Y
)
X 
T
 (σ(Xβ 
(j)
 )−Y)
Then we can finally use gradient descent in order to iteratively update our parameters with respect to a particular class:

β
t
+
1
(
j
)
=
β
t
(
j
)
−
η
[
X
T
(
σ
(
X
β
t
(
j
)
)
−
Y
)
]
β 
t+1
(j)
​
 =β 
t
(j)
​
 −η[X 
T
 (σ(Xβ 
t
(j)
​
 )−Y)]


"""
import numpy as np

def train_softmaxreg(X: np.ndarray, y: np.ndarray, learning_rate: float, iterations: int):
    """
    Gradient-descent training algorithm for Softmax regression, optimizing parameters with Cross Entropy loss.
    Returns optimized coefficients and loss values over iterations.
    """
    n_samples, n_features = X.shape
    n_classes = len(np.unique(y))
    
    # One-hot encode labels
    Y = np.zeros((n_samples, n_classes))
    Y[np.arange(n_samples), y] = 1
    
    # Initialize weights
    W = np.zeros((n_features, n_classes))
    
    losses = []
    
    for _ in range(iterations):
        # Compute logits
        logits = X @ W
        
        # Softmax probabilities
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))  # stability
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        
        # Cross entropy loss
        loss = -np.sum(Y * np.log(probs + 1e-15)) / n_samples
        losses.append(round(loss, 4))
        
        # Gradient
        grad = (X.T @ (probs - Y)) / n_samples
        
        # Update weights
        W -= learning_rate * grad
    
    # Round results
    W_rounded = np.round(W, 4).tolist()
    
    return W_rounded, losses
