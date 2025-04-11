"""
In machine learning and statistics, the softmax function is a generalization of the logistic function that converts a vector of scores into probabilities. The log-softmax function is the logarithm of the softmax function, and it is often used for numerical stability when computing the softmax of large numbers.

Given a 1D numpy array of scores, implement a Python function to compute the log-softmax of the array.

Example:
Input:
A = np.array([1, 2, 3])
print(log_softmax(A))
Output:
array([-2.4076, -1.4076, -0.4076])
Reasoning:
The log-softmax function is applied to the input array [1, 2, 3]. The output array contains the log-softmax values for each element.

"""
"""
Understanding Log Softmax Function
The log softmax function is a numerically stable way of calculating the logarithm of the softmax function. The softmax function converts a vector of arbitrary values (logits) into a vector of probabilities, where each value lies between 0 and 1, and the values sum to 1.

Softmax Function
The softmax function is given by:

softmax
(
x
i
)
=
e
x
i
∑
j
=
1
n
e
x
j
softmax(x 
i
​
 )= 
∑ 
j=1
n
​
 e 
x 
j
​
 
 
e 
x 
i
​
 
 
​
 
Log Softmax Function
Directly applying the logarithm to the softmax function can lead to numerical instability, especially when dealing with large numbers. To prevent this, we use the log-softmax function, which incorporates a shift by subtracting the maximum value from the input vector:

log softmax
(
x
i
)
=
x
i
−
max
⁡
(
x
)
−
log
⁡
(
∑
j
=
1
n
e
x
j
−
max
⁡
(
x
)
)
log softmax(x 
i
​
 )=x 
i
​
 −max(x)−log( 
j=1
∑
n
​
 e 
x 
j
​
 −max(x)
 )
This formulation helps to avoid overflow issues that can occur when exponentiating large numbers. The log-softmax function is particularly useful in machine learning for calculating probabilities in a stable manner, especially when used with cross-entropy loss functions.

"""
import numpy as np

def log_softmax(scores: list) -> np.ndarray:
    # Subtract the maximum value for numerical stability
    scores = scores - np.max(scores)
    return scores - np.log(np.sum(np.exp(scores)))
