"""
Task: Implement the Self-Attention Mechanism
Your task is to implement the self-attention mechanism, which is a fundamental component of transformer models, widely used in natural language processing and computer vision tasks. The self-attention mechanism allows a model to dynamically focus on different parts of the input sequence when generating a contextualized representation.

Your function should return the self-attention output as a numpy array.

Example:
Input:
import numpy as np

X = np.array([[1, 0], [0, 1]])
W_q = np.array([[1, 0], [0, 1]])
W_k = np.array([[1, 0], [0, 1]])
W_v = np.array([[1, 2], [3, 4]])

Q, K, V = compute_qkv(X, W_q, W_k, W_v)
output = self_attention(Q, K, V)

print(output)
Output:
# [[1.660477 2.660477]
#  [2.339523 3.339523]]
Reasoning:
The self-attention mechanism calculates the attention scores for each input, determining how much focus to put on other inputs when generating a contextualized representation. The output is the weighted sum of the values based on the attention scores.

"""
"""
Self-Attention Mechanism
The self-attention mechanism is a fundamental concept in transformer models and is widely used in natural language processing (NLP) and computer vision (CV). It allows models to dynamically weigh different parts of the input sequence, enabling them to capture long-range dependencies effectively.

Understanding Self-Attention
Self-attention helps a model determine which parts of an input sequence are relevant to each other. Instead of treating every word or token equally, self-attention assigns different weights to different parts of the sequence, allowing the model to capture contextual relationships.

For example, in machine translation, self-attention allows the model to focus on relevant words from the input sentence when generating each word in the output.

Mathematical Formulation of Self-Attention
Given an input sequence 
X
X, self-attention computes three key components:

Query (
Q
Q): Represents the current token we are processing.
Key (
K
K): Represents each token in the sequence.
Value (
V
V): Contains the actual token embeddings.
The Query, Key, and Value matrices are computed as:

Q
=
X
W
Q
,
K
=
X
W
K
,
V
=
X
W
V
Q=XW 
Q
​
 ,K=XW 
K
​
 ,V=XW 
V
​
 
where 
W
Q
W 
Q
​
 , 
W
K
W 
K
​
 , and 
W
V
W 
V
​
  are learned weight matrices.

The attention scores are computed using the scaled dot-product attention:

Attention
(
Q
,
K
,
V
)
=
softmax
(
Q
K
T
d
k
)
V
Attention(Q,K,V)=softmax( 
d 
k
​
 
​
 
QK 
T
 
​
 )V
where 
d
k
d 
k
​
  is the dimensionality of the key vectors.

Why Self-Attention is Powerful?
Captures long-range dependencies: Unlike RNNs, which process input sequentially, self-attention can relate any word in the sequence to any other word, regardless of distance.
Parallelization: Since self-attention is computed simultaneously across the entire sequence, it is much faster than sequential models like LSTMs.
Contextual Understanding: Each token is contextually enriched by attending to relevant tokens in the sequence.
Example Calculation
Consider an input sequence of three tokens:

X
=
[
x
1
x
2
x
3
]
X= 
​
  
x 
1
​
 
x 
2
​
 
x 
3
​
 
​
  
​
 
We compute 
Q
Q, 
K
K, and 
V
V as:

Q
=
X
W
Q
,
K
=
X
W
K
,
V
=
X
W
V
Q=XW 
Q
​
 ,K=XW 
K
​
 ,V=XW 
V
​
 
Next, we compute the attention scores:

S
=
Q
K
T
d
k
S= 
d 
k
​
 
​
 
QK 
T
 
​
 
Applying the softmax function:

A
=
softmax
(
S
)
A=softmax(S)
Finally, the weighted sum of values:

Output
=
A
V
Output=AV
Applications of Self-Attention
Self-attention is widely used in:

Transformer models (e.g., BERT, GPT-3) for language modeling.
Speech processing models for transcribing audio.
Vision Transformers (ViTs) for computer vision tasks.
Recommender systems for learning item-user relationships.
Mastering self-attention is essential for understanding modern deep learning architectures, especially in NLP and computer vision.

"""
import numpy as np

def compute_qkv(X, W_q, W_k, W_v):
    Q = np.dot(X, W_q)
    K = np.dot(X, W_k)
    V = np.dot(X, W_v)
    return Q, K, V

def self_attention(Q, K, V):
    d_k = Q.shape[1]
    scores = np.matmul(Q, K.T) / np.sqrt(d_k)
    attention_weights = np.exp(scores) / np.sum(np.exp(scores), axis=1, keepdims=True)
    attention_output = np.matmul(attention_weights, V)
    return attention_output

