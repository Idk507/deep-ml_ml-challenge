"""
Implement the multi-head attention mechanism, a critical component of transformer models. Given Query (Q), Key (K), and Value (V) matrices, compute the attention outputs for multiple heads and concatenate the results.

Example:
Input:
Q = np.array([[1, 0], [0, 1]]), K = np.array([[1, 0], [0, 1]]), V = np.array([[1, 0], [0, 1]]), n_heads = 2
Output:
[[1., 0.], [0., 1.]]
Reasoning:
Multi-head attention is computed for 2 heads using the input Q, K, and V matrices. The resulting outputs for each head are concatenated to form the final attention output.

"""

"""
Understanding Multi-Head Attention
Multi-head attention is a fundamental mechanism in transformer models, allowing the model to focus on different parts of the input sequence simultaneously. This enables the model to capture a wider variety of relationships and dependencies, which is crucial for handling complex data, such as natural language. By using multiple attention heads, the model learns to attend to various aspects of the input at different levels of abstraction, enhancing its ability to capture complex relationships.

Concepts
The attention mechanism allows the model to weigh the importance of different input elements based on their relevance to a specific task. In tasks like machine translation, for example, attention helps the model focus on relevant words in a sentence to understand the overall meaning. Multi-head attention extends this concept by using multiple attention heads, each learning different representations of the input data, which improves the model's ability to capture richer relationships and dependencies.

The process of multi-head attention involves several key steps:

Computing Attention Scores: This involves calculating how much focus each element in the input should receive based on its relationship with other elements.
Applying Softmax: The attention scores are transformed into probabilities using the softmax function, which normalizes the scores so that they sum to one.
Aggregating Results: The final output is computed by taking a weighted sum of the input values, where the weights are determined by the attention scores.
Structure of Multi-Head Attention
The attention mechanism can be described with Query (Q), Key (K), and Value (V) matrices. The process of multi-head attention works by repeating the standard attention mechanism multiple times in parallel, with different sets of learned weight matrices for each attention head.

1. Splitting Q, K, and V
Assume that the input Query (Q), Key (K), and Value (V) matrices have dimensions 
(
seqLen
,
d
m
o
d
e
l
)
(seqLen,d 
model
​
 ), where 
d
model
d 
model
​
  is the model dimension. In multi-head attention, these matrices are divided into n smaller matrices, each corresponding to a different attention head. Each smaller matrix has dimensions 
(
seqLen
,
d
k
)
(seqLen,d 
k
​
 ), where 
d
k
=
d
model
n
d 
k
​
 = 
n
d 
model
​
 
​
  is the dimensionality of each head.

For each attention 
head
i
head 
i
​
 , we get its subset of Query 
Q
i
Q 
i
​
 , Key 
K
i
K 
i
​
 , and Value 
V
i
V 
i
​
 . These subsets are computed independently for each head.

2. Computing Attention for Each Head
Each head independently computes its attention output. The calculation is similar to the single-head attention mechanism:

score
i
=
Q
i
K
i
T
d
k
score 
i
​
 = 
d 
k
​
 
​
 
Q 
i
​
 K 
i
T
​
 
​
 
Where 
d
k
d 
k
​
  is the dimensionality of the key space for each head. The scaling factor 
1
d
k
d 
k
​
 
​
 
1
​
  ensures the dot product doesn't grow too large, preventing instability in the softmax function.

The softmax function is applied to the scores to normalize them, transforming them into attention weights for each head:

SoftmaxScore
i
=
softmax
(
score
i
)
SoftmaxScore 
i
​
 =softmax(score 
i
​
 )
3. Softmax Calculation and Numerical Stability
When computing the softmax function, especially in the context of attention mechanisms, there's a risk of numerical overflow or underflow, which can occur when the attention scores become very large or very small. This issue arises because the exponential function 
exp
⁡
exp grows very quickly, and when dealing with large numbers, it can result in values that are too large for the computer to handle, leading to overflow errors.

To prevent this, we apply a common technique: subtracting the maximum score from each attention score before applying the exponential function. This helps to ensure that the largest value in the attention scores becomes zero, reducing the likelihood of overflow. Here's how it's done:

SoftmaxScore
=
exp
⁡
(
score
−
score
max
)
∑
exp
⁡
(
score
−
score
max
)
SoftmaxScore= 
∑exp(score−score 
max
​
 )
exp(score−score 
max
​
 )
​
 
Where 
score
i
,
max
score 
i,max
​
  is the maximum value of the attention scores for the (i)-th head. Subtracting the maximum score from each individual score ensures that the largest value becomes 0, which prevents the exponentials from becoming too large.

This subtraction does not affect the final result of the softmax calculation because the softmax is a relative function it's the ratios of the exponentials that matter. Therefore, this adjustment ensures numerical stability while maintaining the correctness of the computation.

To summarize, when computing softmax in multi-head attention:

Subtract the maximum score from each attention score before applying the exponential function.
This technique prevents overflow by ensuring that the largest value becomes 0, which keeps the exponential values within a manageable range.
The relative relationships between the scores remain unchanged, so the softmax output remains correct.
By applying this numerical stability trick, the softmax function becomes more robust and prevents computational issues that could arise during training or inference, especially when dealing with large models or sequences.

Finally, the attention output for each 
head
i
head 
i
​
  is computed as:

head
i
=
SoftmaxScore
i
⋅
V
i
head 
i
​
 =SoftmaxScore 
i
​
 ⋅V 
i
​
 
4. Concatenation and Linear Transformation
After computing the attention output for each head, the outputs are concatenated along the feature dimension. This results in a matrix of dimensions 
(
seqLen
,
d
model
)
(seqLen,d 
model
​
 ), where the concatenated attention outputs are passed through a final linear transformation to obtain the final multi-head attention output.

MultiHeadOutput
=
concat
(
head
1
,
head
2
,
…
,
head
n
)
MultiHeadOutput=concat(head 
1
​
 ,head 
2
​
 ,…,head 
n
​
 )
The concatenated result is then linearly transformed using a weight matrix 
W
o
W 
o
​
  to obtain the final output. However, in our case, obtaining the multi-head attention output without this final transformation is sufficient:

MultiHeadOutput
=
W
o
⋅
MultiHeadOutput
MultiHeadOutput=W 
o
​
 ⋅MultiHeadOutput
Key Points
Each attention head processes the input independently using its own set of learned weights. This allows each head to focus on different relationships in the data.
Each head calculates its attention scores based on its corresponding Query, Key, and Value matrices, producing different attention outputs.
The outputs of all attention heads are concatenated to form a unified representation. This concatenated result is then linearly transformed to generate the final output.
Multi-head attention allows the model to attend to different aspects of the input sequence in parallel, making it more capable of learning complex and diverse relationships. This parallelization of attention heads enhances the model's ability to understand the data from multiple angles simultaneously, contributing to improved performance in tasks like machine translation, text generation, and more.

"""

import numpy as np
from typing import Tuple, List

def compute_qkv(X: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the Query (Q), Key (K), and Value (V) matrices.
    
    Args:
    X: numpy array of shape (seq_len, d_model), input sequence
    W_q, W_k, W_v: numpy arrays of shape (d_model, d_model), weight matrices for Q, K, and V
    
    Returns:
    Q, K, V: numpy arrays of shape (seq_len, d_model)
    """
    Q = np.dot(X, W_q)  # Compute the Query matrix Q
    K = np.dot(X, W_k)  # Compute the Key matrix K
    V = np.dot(X, W_v)  # Compute the Value matrix V
    return Q, K, V

def self_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
    """
    Compute self-attention for a single head.
    
    Args:
    Q: numpy array of shape (seq_len, d_k), Query matrix
    K: numpy array of shape (seq_len, d_k), Key matrix
    V: numpy array of shape (seq_len, d_k), Value matrix
    
    Returns:
    attention_output: numpy array of shape (seq_len, d_k), output of the self-attention mechanism
    """
    d_k = Q.shape[1]  # Get the dimension of the keys
    scores = np.matmul(Q, K.T) / np.sqrt(d_k)  # Compute scaled dot-product attention scores
    score_max = np.max(scores, axis=1, keepdims=True)  # Find the maximum score for numerical stability
    attention_weights = np.exp(scores - score_max) / np.sum(np.exp(scores - score_max), axis=1, keepdims=True)  # Compute softmax to get attention weights
    attention_output = np.matmul(attention_weights, V)  # Compute the final attention output
    return attention_output

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, n_heads: int) -> np.ndarray:
    """
    Compute multi-head attention.
    
    Args:
    Q, K, V: numpy arrays of shape (seq_len, d_model), Query, Key, and Value matrices
    n_heads: int, number of attention heads
    
    Returns:
    attention_output: numpy array of shape (seq_len, d_model), final attention output
    """
    d_model = Q.shape[1]  # Get the model dimension
    assert d_model % n_heads == 0  # Ensure d_model is divisible by n_heads
    d_k = d_model // n_heads  # Dimension for each head

    # Reshape Q, K, V to separate heads
    Q_reshaped = Q.reshape(Q.shape[0], n_heads, d_k).transpose(1, 0, 2)  # Reshape and transpose to (n_heads, seq_len, d_k)
    K_reshaped = K.reshape(K.shape[0], n_heads, d_k).transpose(1, 0, 2)  # Reshape and transpose to (n_heads, seq_len, d_k)
    V_reshaped = V.reshape(V.shape[0], n_heads, d_k).transpose(1, 0, 2)  # Reshape and transpose to (n_heads, seq_len, d_k)

    # Compute attention scores for each head
    attentions = []  # Store attention outputs for each head

    for i in range(n_heads):
        attn = self_attention(Q_reshaped[i], K_reshaped[i], V_reshaped[i])  # Compute attention for the i-th head
        attentions.append(attn)  # Collect attention output

    # Concatenate all head outputs
    attention_output = np.concatenate(attentions, axis=-1)  # Concatenate along the last axis (columns)
    return attention_output  # Return the final attention output
