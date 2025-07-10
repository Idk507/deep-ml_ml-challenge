"""
Implement a function that computes the average cross-entropy loss for a batch of predictions in a multi-class classification task. Your function should take in a batch of predicted probabilities and one-hot encoded true labels, then return the average cross-entropy loss. Ensure that you handle numerical stability by clipping probabilities by epsilon.

Example:
Input:
predicted_probs = [[0.7, 0.2, 0.1], [0.3, 0.6, 0.1]]
true_labels = [[1, 0, 0], [0, 1, 0]]
Output:
0.4338
Reasoning:
The predicted probabilities for the correct classes are 0.7 and 0.6. The cross-entropy is computed as -mean(log(0.7), log(0.6)), resulting in approximately 0.4463.
Multi-class Cross-Entropy Loss Implementation
Cross-entropy loss, also known as log loss, measures the performance of a classification model whose output is a probability value between 0 and 1. For multi-class classification tasks, we use the categorical cross-entropy loss.

Mathematical Background
For a single sample with C classes, the categorical cross-entropy loss is defined as:

L
=
−
∑
c
=
1
C
y
c
log
⁡
(
p
c
)
L=−∑ 
c=1
C
​
 y 
c
​
 log(p 
c
​
 )

where:

y
c
y 
c
​
  is a binary indicator (0 or 1) if class label c is the correct classification for the sample
p
c
p 
c
​
  is the predicted probability that the sample belongs to class c
C
C is the number of classes
Implementation Requirements
Your task is to implement a function that computes the average cross-entropy loss across multiple samples:

L
b
a
t
c
h
=
−
1
N
∑
n
=
1
N
∑
c
=
1
C
y
n
,
c
log
⁡
(
p
n
,
c
)
L 
batch
​
 =− 
N
1
​
 ∑ 
n=1
N
​
 ∑ 
c=1
C
​
 y 
n,c
​
 log(p 
n,c
​
 )

where N is the number of samples in the batch.

Important Considerations
Handle numerical stability by adding a small epsilon to avoid log(0)
Ensure predicted probabilities sum to 1 for each sample
Return average loss across all samples
Handle invalid inputs appropriately
The function should take predicted probabilities and true labels as input and return the average cross-entropy loss

"""
import numpy as np

def compute_cross_entropy_loss(predicted_probs: np.ndarray, true_labels: np.ndarray,epsilon = 1e-15) -> float:

    predicted_probs = np.clip(predicted_probs, epsilon, 1 - epsilon)

    #Write your code here
    log_probs = np.log(predicted_probs)
    loss = -np.sum(true_labels * log_probs, axis=1)
    return float(np.mean(loss))
