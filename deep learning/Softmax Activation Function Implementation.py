"""
Write a Python function that computes the softmax activation for a given list of scores. The function should return the softmax values as a list, each rounded to four decimal places.

Example:
Input:
scores = [1, 2, 3]
Output:
[0.0900, 0.2447, 0.6652]
Reasoning:
The softmax function converts a list of values into a probability distribution. The probabilities are proportional to the exponential of each element divided by the sum of the exponentials of all elements in the list.

"""
"""
Understanding the Softmax Activation Function
The softmax function is a generalization of the sigmoid function and is used in the output layer of a neural network model that handles multi-class classification tasks.

Mathematical Definition
The softmax function is mathematically represented as:

softmax
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
e
z
j
softmax(z 
i
​
 )= 
∑ 
j
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
 
Characteristics
Output Range: Each output value is between 0 and 1, and the sum of all outputs is 1.
Probability Distribution: It transforms scores into probabilities, making them easier to interpret and useful for classification tasks.
The softmax function is essential for models where the output needs to represent a probability distribution across multiple classes.

"""

import math
import numpy as np
def softmax(scores: list[float]) -> list[float]:
        e_x = np.exp(scores- np.max(scores))
        return e_x /e_x.sum(axis=0)
