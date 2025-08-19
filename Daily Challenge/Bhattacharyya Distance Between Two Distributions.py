"""


Implement a function to calculate the Bhattacharyya distance between two probability distributions. The function should take two lists representing discrete probability distributions p and q, and return the Bhattacharyya distance rounded to 4 decimal places. If the inputs have different lengths or are empty, return 0.0.

Example:
Input:
p = [0.1, 0.2, 0.3, 0.4], q = [0.4, 0.3, 0.2, 0.1]
Output:
0.1166
Reasoning:
The Bhattacharyya coefficient is calculated as the sum of element-wise square roots of the product of p and q, giving BC = 0.8898. The distance is then -log(0.8898) = 0.1166.

Understanding Bhattacharyya Distance
Bhattacharyya Distance (BD) is a concept in statistics used to measure the similarity or overlap between two probability distributions P(x) and Q(x) on the same domain x.

This differs from KL Divergence, which measures the loss of information when projecting one probability distribution onto another (reference distribution).

Bhattacharyya Distance Formula
The Bhattacharyya distance is defined as:

B
C
(
P
,
Q
)
=
∑
P
(
X
)
⋅
Q
(
X
)
BC(P,Q)=∑ 
P(X)⋅Q(X)
​
 
B
D
(
P
,
Q
)
=
−
ln
⁡
(
B
C
(
P
,
Q
)
)
BD(P,Q)=−ln(BC(P,Q))
where BC (P, Q) is the Bhattacharyya coefficient.

Key Properties
BD is always non-negative:
B
D
≥
0
BD≥0
Symmetric in nature:
B
D
(
P
,
Q
)
=
B
D
(
Q
,
P
)
BD(P,Q)=BD(Q,P)
Applications:
Risk assessment
Stock predictions
Feature scaling
Classification problems
Example Calculation
Consider two probability distributions P(x) and Q(x):

P
(
x
)
=
[
0.1
,
0.2
,
0.3
,
0.4
]
,
Q
(
x
)
=
[
0.4
,
0.3
,
0.2
,
0.1
]
P(x)=[0.1,0.2,0.3,0.4],Q(x)=[0.4,0.3,0.2,0.1]
Bhattacharyya Coefficient:
B
C
(
P
,
Q
)
=
∑
P
(
X
)
⋅
Q
(
X
)
=
0.8898
BC(P,Q)=∑ 
P(X)⋅Q(X)
​
 =0.8898
Bhattacharyya Distance:
B
D
(
P
,
Q
)
=
−
ln
⁡
(
B
C
(
P
,
Q
)
)
=
−
ln
⁡
(
0.8898
)
=
0.1166
BD(P,Q)=−ln(BC(P,Q))=−ln(0.8898)=0.1166
This illustrates how BD quantifies the overlap between two probability distributions.

"""

import numpy as np

def bhattacharyya_distance(p: list[float], q: list[float]) -> float:
    # Your code here
    p = np.array(p)
    q = np.array(q)
    if p.shape != q.shape :
        return 0.0
    p = p / np.sum(p)
    q = q / np.sum(q)
    bc = np.sum(np.sqrt(p*q))
    bd = -np.log(bc)
    return round(bd,4)
