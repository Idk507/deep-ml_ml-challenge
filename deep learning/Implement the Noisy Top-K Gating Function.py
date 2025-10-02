"""
Implement the Noisy Top-K gating mechanism used in Mixture-of-Experts (MoE) models. Given an input matrix, weight matrices, pre-sampled noise, and a sparsity constraint k, compute the final gating probabilities matrix.

Example:
Input:
X = [[1.0, 2.0]]
W_g = [[1.0, 0.0], [0.0, 1.0]]
W_noise = [[0.5, 0.5], [0.5, 0.5]]
N = [[1.0, -1.0]]
k = 2
Output:
[[0.917, 0.0825]]
Reasoning:
This example demonstrates that the gating function produces a sparse softmax output, favoring the higher gate after noise perturbation.

Noisy Top-K Gating
Noisy Top-K Gating is a sparse selection mechanism used in Mixture-of-Experts (MoE) models. It routes input tokens to a subset of available experts, enhancing efficiency and model capacity.

Overview
The core idea is to add learned noise to the gating logits and then select only the top-k experts for each input. This encourages exploration and helps balance load across experts.

Step-by-Step Breakdown
Compute Raw Gate Scores
First, compute two linear projections of the input:

H
base
=
X
W
g
H 
base
​
 =XW 
g
​
 
H
noise
=
X
W
noise
H 
noise
​
 =XW 
noise
​
 
Apply Noise with Softplus Scaling
Add pre-sampled Gaussian noise, scaled by a softplus transformation:

H
=
H
base
+
N
⊙
Softplus
(
H
noise
)
H=H 
base
​
 +N⊙Softplus(H 
noise
​
 )
Top-K Masking
Keep only the top-k elements in each row (i.e., per input), setting the rest to 
−
∞
−∞:

H
′
=
TopK
(
H
,
k
)
H 
′
 =TopK(H,k)
Softmax Over Top-K
Normalize the top-k scores into a valid probability distribution:

G
=
Softmax
(
H
′
)
G=Softmax(H 
′
 )
Worked Example
Let:

X
=
[
[
1.0
,
2.0
]
]
X=[[1.0,2.0]]
W
g
=
[
[
1.0
,
0.0
]
,
[
0.0
,
1.0
]
]
W 
g
​
 =[[1.0,0.0],[0.0,1.0]]
W
noise
=
[
[
0.5
,
0.5
]
,
[
0.5
,
0.5
]
]
W 
noise
​
 =[[0.5,0.5],[0.5,0.5]]
N
=
[
[
1.0
,
−
1.0
]
]
N=[[1.0,−1.0]]
k
=
2
k=2
Step-by-step:

H
base
=
[
1.0
,
2.0
]
H 
base
​
 =[1.0,2.0]
H
noise
=
[
1.5
,
1.5
]
H 
noise
​
 =[1.5,1.5]
Softplus
(
H
noise
)
≈
[
1.804
,
1.804
]
Softplus(H 
noise
​
 )≈[1.804,1.804]
H
=
[
1.0
+
1.804
,
2.0
−
1.804
]
=
[
2.804
,
0.196
]
H=[1.0+1.804,2.0−1.804]=[2.804,0.196]
Softmax over these gives: 
[
0.917
,
0.0825
]
[0.917,0.0825]
Benefits
Computational Efficiency: Activates only k experts per input.
Load Balancing: Injected noise encourages diversity in expert selection.
Improved Generalization: Acts as a regularizer via noise-based gating.
This technique is used in large sparse models like GShard and Switch Transformers.



"""

import numpy as np

def noisy_topk_gating(
    X: np.ndarray,
    W_g: np.ndarray,
    W_noise: np.ndarray,
    N: np.ndarray,
    k: int
) -> np.ndarray:
    """
    Args:
        X: Input data, shape (batch_size, features)
        W_g: Gating weight matrix, shape (features, num_experts)
        W_noise: Noise weight matrix, shape (features, num_experts)
        N: Noise samples, shape (batch_size, num_experts)
        k: Number of experts to keep per example
    Returns:
        Gating probabilities, shape (batch_size, num_experts)
    """
    # Step 1: Compute base and noise projections
    H_base = np.dot(X, W_g)                     # shape: (batch_size, num_experts)
    H_noise = np.dot(X, W_noise)                # shape: (batch_size, num_experts)

    # Step 2: Apply softplus to noise projection
    H_noise_scaled = np.log1p(np.exp(H_noise))  # softplus(x) = log(1 + exp(x))

    # Step 3: Add scaled noise to base
    H = H_base + N * H_noise_scaled             # element-wise multiplication

    # Step 4: Top-K masking
    H_masked = np.full_like(H, -np.inf)
    topk_indices = np.argpartition(-H, k-1, axis=1)[:, :k]  # get indices of top-k values
    for i in range(H.shape[0]):
        H_masked[i, topk_indices[i]] = H[i, topk_indices[i]]

    # Step 5: Softmax over top-k
    max_vals = np.max(H_masked, axis=1, keepdims=True)
    exp_vals = np.exp(H_masked - max_vals)
    sum_exp = np.sum(exp_vals, axis=1, keepdims=True)
    G = exp_vals / sum_exp

    return G
