"""
KL Divergence Estimator for GRPO
Easy
Reinforcement Learning

Implement the unbiased KL divergence estimator used in GRPO (Group Relative Policy Optimization). This estimator computes the KL divergence between the current policy and a reference policy for each sample, which is then used as a regularization term to prevent the policy from deviating too far from the reference.

Example:
Input:
pi_theta = np.array([0.8]), pi_ref = np.array([0.4])
Output:
np.array([0.1931])
Reasoning:
ratio = 0.4/0.8 = 0.5. KL = 0.5 - log(0.5) - 1 = 0.5 - (-0.693) - 1 = 0.193. This penalizes the policy for assigning higher probability than the reference.

KL Divergence Estimator in GRPO
Why KL Divergence?
In reinforcement learning for LLMs, we want to improve the policy while keeping it close to a reference policy. This prevents the model from:

Collapsing to degenerate solutions
Forgetting useful behaviors from pretraining
Exploiting reward hacking strategies
The Standard KL Divergence
The standard KL divergence between two distributions is:

D
K
L
(
P
∣
∣
Q
)
=
∑
x
P
(
x
)
log
⁡
P
(
x
)
Q
(
x
)
D 
KL
​
 (P∣∣Q)=∑ 
x
​
 P(x)log 
Q(x)
P(x)
​
 

However, computing this exactly requires summing over all possible outputs, which is intractable for language models.

The GRPO Estimator
GRPO uses an unbiased per-sample estimator:

D
K
L
(
π
θ
∣
∣
π
r
e
f
)
=
π
r
e
f
(
o
∣
q
)
π
θ
(
o
∣
q
)
−
log
⁡
π
r
e
f
(
o
∣
q
)
π
θ
(
o
∣
q
)
−
1
D 
KL
​
 (π 
θ
​
 ∣∣π 
ref
​
 )= 
π 
θ
​
 (o∣q)
π 
ref
​
 (o∣q)
​
 −log 
π 
θ
​
 (o∣q)
π 
ref
​
 (o∣q)
​
 −1

Let 
r
=
π
r
e
f
π
θ
r= 
π 
θ
​
 
π 
ref
​
 
​
 , then:

D
K
L
=
r
−
log
⁡
(
r
)
−
1
D 
KL
​
 =r−log(r)−1

Properties of This Estimator
Always non-negative: Since 
r
−
log
⁡
(
r
)
−
1
≥
0
r−log(r)−1≥0 for all 
r
>
0
r>0 (with equality only at 
r
=
1
r=1)

Zero when policies match: When 
π
θ
=
π
r
e
f
π 
θ
​
 =π 
ref
​
 , we have 
r
=
1
r=1, so 
D
K
L
=
1
−
0
−
1
=
0
D 
KL
​
 =1−0−1=0

Unbiased: 
E
o
∼
π
θ
[
r
−
log
⁡
(
r
)
−
1
]
E 
o∼π 
θ
​
 
​
 [r−log(r)−1] equals the true KL divergence

Per-sample: Can be computed for each sampled output without summing over all possibilities

Role in GRPO Objective
The full GRPO objective includes this KL term:

J
G
R
P
O
=
E
[
min
⁡
(
ρ
A
,
clip
(
ρ
,
1
−
ϵ
,
1
+
ϵ
)
A
)
−
β
D
K
L
]
J 
GRPO
​
 =E[min(ρA,clip(ρ,1−ϵ,1+ϵ)A)−βD 
KL
​
 ]

The 
β
β hyperparameter controls how strongly to penalize deviation from the reference policy.


"""

import numpy as np

def kl_divergence_estimator(pi_theta: np.ndarray, pi_ref: np.ndarray) -> np.ndarray:
    """
    Compute the unbiased KL divergence estimator used in GRPO.
    
    Formula: D_KL = (pi_ref / pi_theta) - log(pi_ref / pi_theta) - 1
    
    Args:
        pi_theta: Current policy probabilities for each sample (numpy array)
        pi_ref: Reference policy probabilities for each sample (numpy array)
        
    Returns:
        Array of KL divergence estimates (one per sample)
    """
    # Avoid division by zero
    eps = 1e-12
    ratio = (pi_ref + eps) / (pi_theta + eps)
    kl = ratio - np.log(ratio) - 1
    return kl
