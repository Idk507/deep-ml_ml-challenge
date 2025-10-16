"""
Implement the Dynamic Tanh (DyT) function, a normalization-free transformation inspired by the Tanh function. DyT replaces layer normalization in Transformer architectures while preserving squashing behavior and enabling stable training.

Example:
Input:
x = np.array([[[0.14115588, 0.00372817, 0.24126647, 0.22183601]]])
gamma = np.ones((4,))
beta = np.zeros((4,))
alpha = 0.5
print(dynamic_tanh(x, alpha, gamma, beta))
Output:
[[[0.0705, 0.0019, 0.1201, 0.1105]]]
Reasoning:
Each element in the input is scaled by alpha, passed through tanh, and then scaled by gamma and shifted by beta. This mimics the squashing behavior of layer normalization without explicitly using statistics.


A new study (https://arxiv.org/pdf/2503.10622) demonstrates that layer normalization, that is ubiquitous in transformers, produces Tanh-like S-shapes. By incorporating a new layer replacement for normalization called "Dynamic Tanh" (DyT for short), Transformers without normalization can match or exceed the performance of their normalized counterparts, mostly without hyperparameter tuning.

Normalization layer
Consider an standard NLP task, where an input 
x
x has a shape of 
(
B
,
T
,
C
)
(B,T,C), where 
B
B is the batch size, 
T
T - number of tokens (sequence length) and 
C
C - embedding dimensions. Then an output of a normalization layer is generally computed as 
n
o
r
m
(
x
)
=
γ
(
x
−
μ
σ
2
+
ε
)
+
β
norm(x)=γ( 
σ 
2
 +ε
​
 
x−μ
​
 )+β, where 
γ
γ and 
β
β are learnable parameters of shape 
(
C
,
)
(C,). Distribution's statistics are calculated as follows: 
μ
k
=
1
B
T
∑
i
B
∑
j
T
x
i
j
μ 
k
​
 = 
BT
1
​
 ∑ 
i
B
​
 ∑ 
j
T
​
 x 
ij
​
 ; 
σ
k
2
=
1
B
T
∑
i
,
j
(
x
i
j
k
−
μ
k
)
2
σ 
k
2
​
 = 
BT
1
​
 ∑ 
i,j
​
 (x 
ijk
​
 −μ 
k
​
 ) 
2
 

Hyperboloic tangent (Tanh)
Tanh function is defined as a ratio: 
t
a
n
h
(
x
)
=
s
i
n
h
(
x
)
c
o
s
h
(
x
)
=
e
x
p
(
x
)
−
e
x
p
(
−
x
)
e
x
p
(
x
)
+
e
x
p
(
−
x
)
tanh(x)= 
cosh(x)
sinh(x)
​
 = 
exp(x)+exp(−x)
exp(x)−exp(−x)
​
 . Essentially the function allows transformation of an arbitrary domain to 
[
−
1
,
1
]
[−1,1].

Dynamic Tanh (DyT)
Turns out that LN (layer normalization) produces different parts of a 
t
a
n
h
(
k
x
)
tanh(kx), where 
k
k controls the curvature of the tanh curve in the center. The smaller the 
k
k, the smoother is the change from 
−
1
−1 to 
1
1. Hence the study proposes a drop-in replacement for LN given an input tensor 
x
x:

D
y
T
(
x
)
=
γ
∗
t
a
n
h
(
α
x
)
+
β
,
DyT(x)=γ∗tanh(αx)+β,
where:

α
α - learnable parameter that allows scaling the input differently based on its range (tokens producing smaller variance produce less smoother curves). Authors suggest a default value of 
0.5
0.5.
γ
,
β
γ,β - learnable parameters, that scale our output based on the input. Authors suggest initializing these vectors with following default values:
γ
γ as all-one vector
β
β as all-zero
Despite not calculating statistics, DyT preserves the "squashing" effect of LN on extreme values in a non-linear fashion, while almost linearly transforming central parts of the input.

"""
import numpy as np

def dynamic_tanh(x: np.ndarray, alpha: float, gamma: float, beta: float) -> list[float]:
    # Your code here
    return gamma * np.tanh(alpha * x) + beta


