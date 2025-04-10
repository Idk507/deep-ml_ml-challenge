"""
Write a Python function that simulates a single neuron with a sigmoid activation function for binary classification, handling multidimensional input features. The function should take a list of feature vectors (each vector representing multiple features for an example), associated true binary labels, and the neuron's weights (one for each feature) and bias as input. It should return the predicted probabilities after sigmoid activation and the mean squared error between the predicted probabilities and the true labels, both rounded to four decimal places.

Example:
Input:
features = [[0.5, 1.0], [-1.5, -2.0], [2.0, 1.5]], labels = [0, 1, 0], weights = [0.7, -0.4], bias = -0.1
Output:
([0.4626, 0.4134, 0.6682], 0.3349)
Reasoning:
For each input vector, the weighted sum is calculated by multiplying each feature by its corresponding weight, adding these up along with the bias, then applying the sigmoid function to produce a probability. The MSE is calculated as the average squared difference between each predicted probability and the corresponding true label.


"""
"""
Single Neuron Model with Multidimensional Input and Sigmoid Activation
This task involves a neuron model designed for binary classification with multidimensional input features, using the sigmoid activation function to output probabilities. It also involves calculating the mean squared error (MSE) to evaluate prediction accuracy.

Mathematical Background
Neuron Output Calculation:

z
=
∑
(
w
e
i
g
h
t
i
×
f
e
a
t
u
r
e
i
)
+
b
i
a
s
z=∑(weight 
i
​
 ×feature 
i
​
 )+bias
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
 
MSE Calculation:

M
S
E
=
1
n
∑
(
p
r
e
d
i
c
t
e
d
−
t
r
u
e
)
2
MSE= 
n
1
​
 ∑(predicted−true) 
2
 
Explanation of Terms
( z ): The sum of weighted inputs plus bias.
( \sigma(z) ): The sigmoid activation output.
( predicted ): The probabilities after sigmoid activation.
( true ): The true binary labels.
Practical Implementation
Each feature vector is processed to calculate a combined weighted sum, which is then passed through the sigmoid function to determine the probability of the input belonging to the positive class.
MSE provides a measure of error, offering insights into the model's performance and aiding in its optimization.

"""


import math
def single_neuron_model(features, labels, weights, bias):
    probabilities = []
    for feature_vector in features:
        z = sum(weight * feature for weight, feature in zip(weights, feature_vector)) + bias
        prob = 1 / (1 + math.exp(-z))
        probabilities.append(round(prob, 4))
    
    mse = sum((prob - label) ** 2 for prob, label in zip(probabilities, labels)) / len(labels)
    mse = round(mse, 4)
    
    return probabilities, mse
