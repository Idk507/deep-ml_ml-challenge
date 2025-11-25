"""
Implement functions to (1) generate smoothed one-hot label distributions and (2) compute cross-entropy loss using those smoothed labels for a multi-class classification problem. The implementation should handle different values of the smoothing parameter epsilon, use numerically stable log-softmax, and support optional rounding of the final loss value.

Example:
Input:
logits = [[2.0, 0.0, -1.0], [0.0, 1.0, 0.0]]
y_true = [0, 2]
loss = label_smoothing_cross_entropy(logits, y_true, num_classes=3, epsilon=0.1, round_decimals=6)
print(loss)
Output:
0.927312
Reasoning:
With ε=0.1 and K=3, the true class gets 0.9333 probability and others 0.0333. Using stable log-softmax and averaging cross-entropy gives ≈ 0.927312.


Understanding Label Smoothing and Cross-Entropy
1. Motivation
In classification tasks, targets are often one-hot encoded, meaning that the true class receives a probability of 1 and all others 0. This can cause overconfidence in neural networks. Label smoothing helps prevent overfitting by making targets slightly less certain.

2. Label Smoothing Definition
For a 
K
K-class classification problem and smoothing parameter 
ε
∈
[
0
,
1
]
ε∈[0,1], the smoothed target distribution for a true class 
y
y is defined as:

t
k
=
{
1
−
ε
+
ε
K
,
if 
k
=
y
,
ε
K
,
if 
k
≠
y
.
t 
k
​
 ={ 
1−ε+ 
K
ε
​
 ,
K
ε
​
 ,
​
  
if k=y,
if k

=y.
​
 
Here, 
ε
ε controls how much smoothing to apply:

ε
=
0
ε=0: Standard one-hot encoding (no smoothing)
ε
=
1
ε=1: Fully uniform distribution
The resulting smoothed vector still sums to 1.

3. Cross-Entropy with Label Smoothing
Given model logits 
z
∈
R
K
z∈R 
K
  and smoothed target 
t
t, the cross-entropy loss is:

L
(
z
,
t
)
=
−
∑
k
=
1
K
t
k
log
⁡
p
k
,
L(z,t)=− 
k=1
∑
K
​
 t 
k
​
 logp 
k
​
 ,
where 
p
k
p 
k
​
  is the modelâs predicted probability for class 
k
k:

p
k
=
e
z
k
∑
j
=
1
K
e
z
j
.
p 
k
​
 = 
∑ 
j=1
K
​
 e 
z 
j
​
 
 
e 
z 
k
​
 
 
​
 .
4. Numerical Stability with Log-Softmax
To avoid overflow when computing 
e
z
k
e 
z 
k
​
 
 , use the log-sum-exp trick:

log
⁡
p
k
=
z
k
−
(
max
⁡
j
z
j
+
log
⁡
∑
j
=
1
K
e
z
j
−
max
⁡
j
z
j
)
.
logp 
k
​
 =z 
k
​
 −( 
j
max
​
 z 
j
​
 +log 
j=1
∑
K
​
 e 
z 
j
​
 −max 
j
​
 z 
j
​
 
 ).
5. Why It Matters
Label smoothing reduces the modelâs overconfidence by distributing a small portion of probability mass across all classes, leading to:

Better calibration of predicted probabilities
Improved generalization
More stable gradients during training

"""
import numpy as np

def smooth_labels(y_true, num_classes, epsilon):
    """
    Create smoothed one-hot target vectors.

    Args:
        y_true: Iterable[int] of shape (N,) with values in [0, K-1]
        num_classes: int, total number of classes (K)
        epsilon: float in [0, 1]

    Returns:
        np.ndarray of shape (N, K) with smoothed probabilities.
    """
    y_true = np.asarray(y_true, dtype=int)
    N = len(y_true)

    # Base distribution: epsilon/K for all classes
    smoothed = np.full((N, num_classes), epsilon / num_classes)

    # Add (1 - epsilon) to the true class
    smoothed[np.arange(N), y_true] += (1.0 - epsilon)

    return smoothed


def label_smoothing_cross_entropy(logits, y_true, num_classes, epsilon=0.1, round_decimals=None):
    """
    Compute mean cross-entropy between logits and smoothed targets using stable log-softmax.

    Args:
        logits: Array-like of shape (N, K), model output scores.
        y_true: Array-like of shape (N,), integer class indices.
        num_classes: int, number of classes (K).
        epsilon: float in [0, 1].
        round_decimals: int | None, round the loss to this many decimals if given.

    Returns:
        float: Mean cross-entropy loss.
    """
    logits = np.asarray(logits, dtype=float)
    y_true = np.asarray(y_true, dtype=int)

    # Get smoothed targets
    targets = smooth_labels(y_true, num_classes, epsilon)

    # Numerically stable log-softmax
    max_logits = np.max(logits, axis=1, keepdims=True)
    log_probs = logits - max_logits - np.log(np.sum(np.exp(logits - max_logits), axis=1, keepdims=True))

    # Cross-entropy: -sum(target * log_probs)
    loss_per_sample = -np.sum(targets * log_probs, axis=1)
    mean_loss = np.mean(loss_per_sample)

    if round_decimals is not None:
        mean_loss = np.round(mean_loss, round_decimals)

    return float(mean_loss)


# Example usage
logits = [[2.0, 0.0, -1.0], [0.0, 1.0, 0.0]]
y_true = [0, 2]
loss = label_smoothing_cross_entropy(logits, y_true, num_classes=3, epsilon=0.1, round_decimals=6)
print(loss)  # Expected output: 0.927312
