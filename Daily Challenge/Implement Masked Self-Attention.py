"""
Implement Masked Self-Attention

Implement masked self-attention, a variation of the attention mechanism used in sequence modeling tasks such as text generation. Your task is to compute masked self-attention using query (Q), key (K), value (V) matrices and an attention mask.

Example:
Input:
masked_attention(Q, K, V, mask)
Output:
[[547. 490. 399. 495. 485. 439. 645. 393.]
 [547. 490. 399. 495. 485. 439. 645. 393.]
 [471. 472. 429. 538. 377. 450. 531. 362.]
 [471. 472. 429. 538. 377. 450. 531. 362.]
 [471. 472. 429. 538. 377. 450. 531. 362.]
 [471. 472. 429. 538. 377. 450. 531. 362.]]
Reasoning:
The function computes self-attention by applying a mask to restrict information flow, ensuring causal dependencies are maintained.
Understanding Masked Attention
Masked attention is a variation of the attention mechanism used primarily in sequence modeling tasks, such as language modeling and text generation. The key idea behind masked attention is to control the flow of information by selectively masking certain elements in the input sequence. This ensures that the model attends only to valid positions when computing attention scores.

Masked attention is particularly useful in autoregressive tasks where future information should not influence the current prediction. By masking out future tokens, the model is constrained to attend only to preceding tokens or the current token, preserving causality during training and inference.

Concepts
The attention mechanism enables the model to weigh the importance of different elements in the input sequence based on their relevance to a specific task. Masked attention modifies this process by incorporating a mask, which defines which elements the model is allowed to attend to. This ensures that the attention mechanism respects temporal or structural constraints, such as the directionality of time in sequence data.

The process of masked attention involves the following steps:

Computing Attention Scores: The model calculates how much focus each element in the sequence should receive based on its relationship with other elements.
Applying the Mask: A mask is applied to restrict attention to specific positions in the sequence. Elements outside the allowed range are effectively ignored.
Normalizing Scores: The masked scores are transformed into probabilities using the softmax function.
Computing the Output: The final output is computed as a weighted sum of the input values, with weights determined by the normalized attention scores.
Structure of Masked Attention
The attention mechanism can be described using Query (Q), Key (K), and Value (V) matrices. In masked attention, these matrices interact with an additional mask to determine the attention distribution.

1. Query, Key, and Value Matrices
Query (Q): Represents the current element for which the model is computing attention.
Key (K): Encodes information about all elements in the sequence.
Value (V): Contains the representations that will be aggregated into the output.
Assume that the input sequence has a length of 
seqLen
seqLen and the model dimension is 
d
model
d 
model
​
 . The dimensions of the Q, K, and V matrices are:

Query (Q): 
(
seqLen
,
d
model
)
(seqLen,d 
model
​
 )
Key (K): 
(
seqLen
,
d
model
)
(seqLen,d 
model
​
 )
Value (V): 
(
seqLen
,
d
model
)
(seqLen,d 
model
​
 )
2. Computing Attention Scores
The raw attention scores are computed as the scaled dot product between the Query (Q) and Key (K) matrices:

score
=
Q
K
T
d
k
score= 
d 
k
​
 
​
 
QK 
T
 
​
 
Where 
d
k
d 
k
​
  is the dimensionality of the key space. The scaling factor 
1
d
k
d 
k
​
 
​
 
1
​
  ensures that the dot product values do not grow excessively large, preventing instability in the softmax function.

3. Applying the Mask
The mask is used to control which elements the model is allowed to attend to. Typically, the mask is a binary matrix of dimensions 
(
seqLen
,
seqLen
)
(seqLen,seqLen), where:

A value of 0 indicates that attention is allowed.
A value of 
−
∞
−∞ (or a very large negative value like 
−
1
e
9
−1e9) indicates that attention is prohibited.
The raw attention scores are modified by adding the mask:

maskedScore
=
score
+
mask
maskedScore=score+mask
This ensures that prohibited positions receive attention scores that are effectively 
−
∞
−∞, making their softmax probabilities zero.

4. Softmax Calculation
The softmax function is applied to the masked scores to compute attention weights. To ensure numerical stability, the maximum score in each row is subtracted before applying the softmax function:

SoftmaxScore
=
exp
⁡
(
maskedScore
−
maskedScore
max
)
∑
exp
⁡
(
maskedScore
−
maskedScore
max
)
SoftmaxScore= 
∑exp(maskedScore−maskedScore 
max
​
 )
exp(maskedScore−maskedScore 
max
​
 )
​
 
5. Computing the Output
The final attention output is computed as a weighted sum of the Value (V) matrix, with weights determined by the attention scores:

output
=
SoftmaxScore
⋅
V
output=SoftmaxScore⋅V
Key Points
Masking Future Tokens: In autoregressive tasks, a triangular mask is used to prevent the model from attending to future positions. For a sequence of length 
n
n, the mask is an upper triangular matrix with 0s in the lower triangle and 
−
∞
−∞ in the upper triangle.

Example:

mask
=
[
0
−
∞
−
∞
0
0
−
∞
0
0
0
]
mask= 
​
  
0
0
0
​
  
−∞
0
0
​
  
−∞
−∞
0
​
  
​
 
Numerical Stability: Subtracting the maximum score before applying softmax ensures numerical stability and prevents overflow or underflow errors.

Flexibility: The mask can be customized to handle other constraints, such as ignoring padding tokens in variable-length sequences.

By selectively controlling the flow of information through masking, masked attention ensures that the model respects temporal or structural constraints, enabling it to generate coherent and contextually accurate outputs in sequence modeling tasks.

"""
import numpy as np

def compute_qkv(X: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray):
	"""
	Compute Query (Q), Key (K), and Value (V) matrices.
	"""
	return np.dot(X, W_q), np.dot(X, W_k), np.dot(X, W_v)

def masked_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, mask: np.ndarray) -> np.ndarray:
	"""
	Compute masked self-attention.
	"""
	score = np.matmul(Q,K.T)
    maskedscore = score + mask
    softmax = np.exp(maskedscore - np.max(maskedscore,axis=1,keepdims=True))
    output = np.matmul(softmax,V)
    return output
