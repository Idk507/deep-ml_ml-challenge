"""
KL Divergence Between Two Normal Distributions

Task: Implement KL Divergence Between Two Normal Distributions
Your task is to compute the Kullback-Leibler (KL) divergence between two normal distributions. KL divergence measures how one probability distribution differs from a second, reference probability distribution.

Write a function kl_divergence_normal(mu_p, sigma_p, mu_q, sigma_q) that calculates the KL divergence between two normal distributions, where ( P \sim N(\mu_P, \sigma_P^2) ) and ( Q \sim N(\mu_Q, \sigma_Q^2) ).

The function should return the KL divergence as a floating-point number.
Example:
Input:
mu_p = 0.0
sigma_p = 1.0
mu_q = 1.0
sigma_q = 1.0

print(kl_divergence_normal(mu_p, sigma_p, mu_q, sigma_q))
Output:
0.5
Reasoning:
The KL divergence between the normal distributions ( P ) and ( Q ) with parameters ( \mu_P = 0.0 ), ( \sigma_P = 1.0 ) and ( \mu_Q = 1.0 ), ( \sigma_Q = 1.0 ) is 0.5.
Understanding Kullback-Leibler Divergence (KL Divergence)
KL Divergence is a key concept in probability theory and information theory, used to measure the difference between two probability distributions. It quantifies how much information is lost when one distribution is used to approximate another.

What is KL Divergence?
KL Divergence is defined as:

D
KL
(
P
∥
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
(
P
(
x
)
Q
(
x
)
)
D 
KL
​
 (P∥Q)= 
x
∑
​
 P(x)log( 
Q(x)
P(x)
​
 )
Where:

P
(
x
)
P(x) is the true probability distribution.
Q
(
x
)
Q(x) is the approximating probability distribution.
The sum is taken over all possible outcomes 
x
x.
Intuition Behind KL Divergence
KL Divergence measures the "extra" number of bits required to code samples from 
P
(
x
)
P(x) using the distribution 
Q
(
x
)
Q(x), instead of using the true distribution 
P
(
x
)
P(x).

If 
P
P and 
Q
Q are identical, 
D
KL
(
P
∥
Q
)
=
0
D 
KL
​
 (P∥Q)=0, meaning no extra bits are needed.
If 
Q
Q is very different from 
P
P, the divergence will be large, indicating a poor approximation.
KL Divergence is always non-negative due to its relationship with the Kullback-Leibler inequality, which is a result of Gibbs' inequality.

Key Properties
Asymmetry: 
D
KL
(
P
∥
Q
)
≠
D
KL
(
Q
∥
P
)
D 
KL
​
 (P∥Q)

=D 
KL
​
 (Q∥P). That is, KL Divergence is not a true distance metric.
Non-negativity: 
D
KL
(
P
∥
Q
)
≥
0
D 
KL
​
 (P∥Q)≥0 for all probability distributions 
P
P and 
Q
Q.
Applicability: KL Divergence is used in various fields, including machine learning, data science, and natural language processing, to compare probability distributions or models.
Example
Consider two discrete probability distributions 
P
(
x
)
P(x) and 
Q
(
x
)
Q(x):

P
(
x
)
=
[
0.4
,
0.6
]
,
Q
(
x
)
=
[
0.5
,
0.5
]
P(x)=[0.4,0.6],Q(x)=[0.5,0.5]
The KL Divergence between these two distributions is calculated as:

D
KL
(
P
∥
Q
)
=
0.4
log
⁡
(
0.4
0.5
)
+
0.6
log
⁡
(
0.6
0.5
)
D 
KL
​
 (P∥Q)=0.4log( 
0.5
0.4
​
 )+0.6log( 
0.5
0.6
​
 )
This gives the divergence measure, quantifying how much information is lost when using 
Q
(
x
)
Q(x) to approximate 
P
(
x
)
P(x).

KL Divergence plays an essential role in fields like machine learning, where it is used for tasks such as model evaluation, anomaly detection, and optimization.
"""
import numpy as np

def kl_divergence_normal(mu_p, sigma_p, mu_q, sigma_q):
	term1 = np.log(sigma_q/sigma_p)
    term2 = (sigma_p ** 2 +(mu_p-mu_q)**2)/(2*(sigma_q**2))
    kl_div = term1 + term2 - 0.5
    return kl_div
