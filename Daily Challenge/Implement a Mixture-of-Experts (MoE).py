"""
Implement a Mixture-of-Experts (MoE) layer using softmax gating and top-k routing. Given an input tensor, a set of expert weight matrices, a gating weight matrix, and parameters specifying the number of experts and the value of k, compute the final MoE output by selecting the top-k experts per token, applying their transformations, and aggregating the results weighted by the normalized gating probabilities.

Example:
Input:
x = np.arange(12).reshape(2, 3, 2)
We = np.ones((4, 2, 2))
Wg = np.ones((2, 4))
top_k = 1
Output:
[[[1, 1], [5, 5], [9, 9]], [[13, 13], [17, 17], [21, 21]]]
Reasoning:
Each token is routed to its top expert and processed using a weight matrix of ones. The result matches the input tokens due to identity transformation and weight 1.

Mixture of Experts Layer
Mixture-of-Experts layers route each token through a small subset of expert networks, reducing computation while retaining flexibility.

1. Gating with Softmax
Logits: For each token 
t
t, compute a vector of gating scores 
g
t
∈
R
E
g 
t
​
 ∈R 
E
 , where 
E
E is the number of experts.
Softmax: Convert scores into a probability distribution
α
t
,
j
=
exp
⁡
(
g
t
,
j
−
max
⁡
j
g
t
,
j
)
∑
j
′
=
1
E
exp
⁡
(
g
t
,
j
′
−
max
⁡
j
g
t
,
j
′
)
.
α 
t,j
​
 = 
∑ 
j 
′
 =1
E
​
 exp(g 
t,j 
′
 
​
 −max 
j
​
 g 
t,j 
′
 
​
 )
exp(g 
t,j
​
 −max 
j
​
 g 
t,j
​
 )
​
 .
2. Top-
k
k Selection
Sparsity: Keep only the 
k
k largest weights per token, zeroing out the rest.
Renormalize: For token 
t
t, let 
K
t
K 
t
​
  be the indices of the top 
k
k experts. Then
α
~
t
,
j
=
{
α
t
,
j
∑
i
∈
K
t
α
t
,
i
j
∈
K
t
,
0
otherwise.
α
~
  
t,j
​
 = 
⎩
⎨
⎧
​
  
∑ 
i∈K 
t
​
 
​
 α 
t,i
​
 
α 
t,j
​
 
​
 
0
​
  
j∈K 
t
​
 ,
otherwise.
​
 
3. Expert Computation
Each expert 
i
i applies its own linear transform to the token embedding 
x
t
x 
t
​
 :

O
t
(
i
)
=
x
t
 
W
e
(
i
)
,
O 
t
(i)
​
 =x 
t
​
 W 
e
(i)
​
 ,
where 
W
e
(
i
)
W 
e
(i)
​
  is the expert's 
d
×
d
d×d weight matrix.

4. Weighted Aggregation
Combine the selected experts' outputs for each token:

y
t
=
∑
i
=
1
E
α
~
t
,
i
 
O
t
(
i
)
.
y 
t
​
 = 
i=1
∑
E
​
  
α
~
  
t,i
​
 O 
t
(i)
​
 .
The result 
y
t
y 
t
​
  lives in the original embedding space 
R
d
R 
d
 .

Example Walk Through
Suppose one sentence of length 2, embedding size 3, 
E
=
4
E=4 experts, and 
k
=
2
k=2.

After flattening, you get 2 softmax distributions of length 4.
You pick the top 2 experts for each token and renormalize their weights.
Each selected expert produces a 3-dimensional output for its tokens.
You weight and sum those outputs to yield the final 3-dimensional vector per token.
This sparse routing mechanism dramatically cuts computation only 
k
k experts run per token instead of all 
E
E while retaining the expressivity of a full ensemble.
"""
import numpy as np

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def get_top_k(arr: np.ndarray, k: int):
    idx = np.argpartition(arr, -k)[..., -k:]
    vals = np.take_along_axis(arr, idx, axis=-1)
    return idx, vals

def expert(x: np.ndarray, We_i: np.ndarray):
    # x: [n_tokens, d_model]
    # We_i: [d_model, d_model]
    return x @ We_i

def gate(x: np.ndarray, Wg: np.ndarray):
    # x: [n_batch * l_seq, d_model]
    # Wg: [n_batch * l_seq, n_experts]
    return x @ Wg

def moe(x: np.ndarray, We: np.ndarray, Wg: np.ndarray, n_experts: int, top_k: int):
    # x: [n_batch, l_seq, d_model]
    # We: [n_experts, d_model, d_model]
    # Wg: [n_batch * l_seq, n_experts]

    n_batch, l_seq, d_model = x.shape

    # flatten batch and sequence dimensions for easier indexing
    # x_flat: [n_batch * l_seq, d_model]
    x_flat = x.reshape(-1, d_model)
    n_tokens, _ = x_flat.shape

    gating_logits = gate(x_flat, Wg)
    gating_weights = softmax(gating_logits, axis=-1)

    topk_idx, topk_weights = get_top_k(gating_weights, top_k)
    topk_idx_flat = topk_idx.flatten()  # [n_tokens * top_k]
    # mapping from top K expert indices to token indices: [n_tokens * top_k]
    token_idx_flat = np.arange(n_tokens).repeat(top_k)

    topk_weights_norm = topk_weights / topk_weights.sum(axis=1, keepdims=True)
    topk_weights_norm_flat = topk_weights_norm.flatten()

    # prepare result memory for aggregation: [n_tokens, d_model]
    output_flat = np.zeros_like(x_flat)
    for i in range(n_experts):
        mask = topk_idx_flat == i
        tokens_expert_i = token_idx_flat[mask]

        if tokens_expert_i.size > 0:
            x_expert_i = x_flat[tokens_expert_i]
            output_expert_i = expert(x_expert_i, We[i, ...])
            output_expert_i *= topk_weights_norm_flat[mask, None]

            # scatter add to result memory
            np.add.at(output_flat, tokens_expert_i, output_expert_i)

    return output_flat.reshape(n_batch, l_seq, d_model)
