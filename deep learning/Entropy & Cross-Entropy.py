"""
Implement functions to compute both entropy and cross-entropy for probability distributions. Entropy measures the uncertainty or information content in a single distribution, while cross-entropy measures the average number of bits needed to identify events from distribution P when using a coding scheme optimized for distribution Q. Return both values as a tuple.

Example:
Input:
P = [0.7, 0.3], Q = [0.6, 0.4]
Output:
(0.610864, 0.632465)
Reasoning:
Entropy: 
H
(
P
)
=
−
(
0.7
log
⁡
(
0.7
)
+
0.3
log
⁡
(
0.3
)
)
≈
0.611
H(P)=−(0.7log(0.7)+0.3log(0.3))≈0.611. Cross-entropy: 
H
(
P
,
Q
)
=
−
(
0.7
log
⁡
(
0.6
)
+
0.3
log
⁡
(
0.4
)
)
≈
0.632
H(P,Q)=−(0.7log(0.6)+0.3log(0.4))≈0.632. Cross-entropy is higher because Q doesn't perfectly match P.

Learn About topic
Understanding Entropy and Cross-Entropy
Entropy and cross-entropy are fundamental concepts in information theory that quantify uncertainty and the cost of encoding information. These measures are essential in machine learning, particularly in classification tasks.

Entropy: Measuring Uncertainty
Entropy measures the average amount of information (or uncertainty) in a probability distribution. For a discrete distribution 
P
P:

H
(
P
)
=
−
∑
i
P
(
i
)
log
⁡
P
(
i
)
H(P)=− 
i
∑
​
 P(i)logP(i)
By convention, 
0
log
⁡
0
=
0
0log0=0 (since 
lim
⁡
x
→
0
x
log
⁡
x
=
0
lim 
x→0
​
 xlogx=0).

Intuition: Entropy answers "How surprised would we be, on average, by events drawn from this distribution?" Higher entropy means more uncertainty.

Key Properties of Entropy
Minimum entropy (0): Occurs for deterministic distributions where one outcome has probability 1:

P
=
[
1
,
0
,
0
,
…
]
  
⟹
  
H
(
P
)
=
0
P=[1,0,0,…]⟹H(P)=0
No surprise - we always know what will happen.

Maximum entropy: For 
n
n outcomes, entropy is maximized by the uniform distribution:

P
=
[
1
n
,
1
n
,
…
,
1
n
]
  
⟹
  
H
(
P
)
=
log
⁡
(
n
)
P=[ 
n
1
​
 , 
n
1
​
 ,…, 
n
1
​
 ]⟹H(P)=log(n)
Maximum surprise - all outcomes equally likely.

Non-negativity: 
H
(
P
)
≥
0
H(P)≥0 always.

Cross-Entropy: Measuring Distribution Difference
Cross-entropy measures the average number of bits needed to encode events from distribution 
P
P using a coding scheme optimized for distribution 
Q
Q:

H
(
P
,
Q
)
=
−
∑
i
P
(
i
)
log
⁡
Q
(
i
)
H(P,Q)=− 
i
∑
​
 P(i)logQ(i)
Intuition: If 
Q
Q is a model's predicted distribution and 
P
P is the true distribution, cross-entropy measures how "surprised" we are on average when using 
Q
Q to encode events that actually follow 
P
P.

Relationship Between Entropy and Cross-Entropy
Cross-entropy is always at least as large as entropy:

H
(
P
,
Q
)
≥
H
(
P
)
H(P,Q)≥H(P)
With equality if and only if 
P
=
Q
P=Q. This follows from Gibbs' inequality.

The difference is the KL divergence:

H
(
P
,
Q
)
=
H
(
P
)
+
D
K
L
(
P
∣
∣
Q
)
H(P,Q)=H(P)+D 
KL
​
 (P∣∣Q)
Where 
D
K
L
(
P
∣
∣
Q
)
≥
0
D 
KL
​
 (P∣∣Q)≥0 measures how different 
Q
Q is from 
P
P.

Example Calculation
Consider 
P
=
[
0.7
,
0.3
]
P=[0.7,0.3] and 
Q
=
[
0.6
,
0.4
]
Q=[0.6,0.4].

Entropy of P:

H
(
P
)
=
−
(
0.7
log
⁡
0.7
+
0.3
log
⁡
0.3
)
H(P)=−(0.7log0.7+0.3log0.3)
=
−
(
0.7
×
(
−
0.357
)
+
0.3
×
(
−
1.204
)
)
=−(0.7×(−0.357)+0.3×(−1.204))
=
0.250
+
0.361
=
0.611
 nats
=0.250+0.361=0.611 nats
Cross-Entropy:

H
(
P
,
Q
)
=
−
(
0.7
log
⁡
0.6
+
0.3
log
⁡
0.4
)
H(P,Q)=−(0.7log0.6+0.3log0.4)
=
−
(
0.7
×
(
−
0.511
)
+
0.3
×
(
−
0.916
)
)
=−(0.7×(−0.511)+0.3×(−0.916))
=
0.358
+
0.275
=
0.632
 nats
=0.358+0.275=0.632 nats
Since 
Q
≠
P
Q

=P, we have 
H
(
P
,
Q
)
>
H
(
P
)
H(P,Q)>H(P). The difference (0.021 nats) is the KL divergence.

Why Cross-Entropy Matters in Machine Learning
Loss Function: In classification, cross-entropy loss measures how well predicted probabilities match true labels:

Loss
=
−
∑
i
y
i
log
⁡
(
y
^
i
)
Loss=− 
i
∑
​
 y 
i
​
 log( 
y
^
​
  
i
​
 )
Where 
y
y is the true distribution (often one-hot) and 
y
^
y
^
​
  is the model's prediction.

Training Objective: Minimizing cross-entropy is equivalent to:

Maximizing log-likelihood
Minimizing KL divergence from true distribution
Maximum likelihood estimation
Binary Classification: For binary outcomes, cross-entropy becomes:

H
(
P
,
Q
)
=
−
[
p
log
⁡
q
+
(
1
−
p
)
log
⁡
(
1
−
q
)
]
H(P,Q)=−[plogq+(1−p)log(1−q)]
This is the binary cross-entropy loss commonly used in neural networks.

Information-Theoretic Interpretation
Entropy as optimal code length: 
H
(
P
)
H(P) represents the minimum average bits needed to encode messages from 
P
P using an optimal code.

Cross-entropy as suboptimal code length: 
H
(
P
,
Q
)
H(P,Q) represents the average bits needed when using a code optimized for 
Q
Q to encode messages from 
P
P.

Penalty for mismatch: The extra cost 
H
(
P
,
Q
)
−
H
(
P
)
=
D
K
L
(
P
∣
∣
Q
)
H(P,Q)−H(P)=D 
KL
​
 (P∣∣Q) quantifies the inefficiency of using the wrong distribution.

Practical Considerations
Logarithm base:

Natural log (base 
e
e): Results in "nats"
Base 2: Results in "bits"
Both are valid; just scale factors
Numerical stability: Add small epsilon when computing 
log
⁡
(
Q
(
i
)
)
log(Q(i)) to avoid 
log
⁡
(
0
)
log(0):

log
⁡
(
Q
(
i
)
+
ϵ
)
where 
ϵ
≈
10
−
10
log(Q(i)+ϵ)where ϵ≈10 
−10
 
Infinite cross-entropy: If 
P
(
i
)
>
0
P(i)>0 but 
Q
(
i
)
=
0
Q(i)=0, cross-entropy is infinite - the coding scheme fails for possible events.

Applications Beyond ML
Data Compression: Huffman coding uses entropy to determine optimal compression.

Model Selection: Lower cross-entropy indicates better predictive models.

Natural Language Processing: Perplexity (
2
H
(
P
,
Q
)
2 
H(P,Q)
 ) measures language model quality.

Physics: Entropy appears in statistical mechanics and thermodynamics.

Neuroscience: Information-theoretic measures quantify neural coding efficiency.


"""

import numpy as np

def entropy_and_cross_entropy(P: list[float], Q: list[float]) -> tuple[float, float]:
	"""
	Compute entropy of P and cross-entropy between P and Q.
	
	Args:
		P: True probability distribution
		Q: Predicted probability distribution
	
	Returns:
		Tuple of (entropy H(P), cross-entropy H(P,Q))
	"""
	P = np.array(P)
	Q = np.array(Q)
	
	# Small epsilon to avoid log(0)
	epsilon = 1e-10
	
	# Compute entropy H(P) = -∑ P(i) * log(P(i))
	# Only compute for non-zero P values (0*log(0) = 0 by convention)
	entropy = 0.0
	for p in P:
		if p > 0:
			entropy -= p * np.log(p)
	
	# Compute cross-entropy H(P,Q) = -∑ P(i) * log(Q(i))
	# Add epsilon to Q to avoid log(0)
	cross_entropy = -np.sum(P * np.log(Q + epsilon))
	
	return (entropy, cross_entropy)
