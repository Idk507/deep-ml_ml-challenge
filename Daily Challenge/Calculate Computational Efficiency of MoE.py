"""
Calculate the computational cost savings of an MoE layer compared to a dense layer, as discussed in the paper 'Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer.' Given the number of experts, sparsity (number of active experts), and input/output dimensions, compute the floating-point operations (FLOPs) for both and determine the savings percentage.

Example:
Input:
compute_efficiency(1000, 2, 512, 512)
Output:
99.8
Reasoning:
Dense layer FLOPs: 1000 * 512 * 512 = 262,144,000. MoE FLOPs: 2 * 512 * 512 = 524,288. Savings: ((262,144,000 - 524,288) / 262,144,000) x 100 ≈ 99.8%.

Understanding MoE Efficiency
The paper "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer" introduces the idea of activating only a few expert networks per input to drastically reduce computation. This is known as conditional computation, and it allows models to scale to billions of parameters without significantly increasing cost.

Key Idea
In a dense layer, every input goes through the full set of parameters.
In a Mixture-of-Experts (MoE) layer, only 
k
k out of 
n
n experts are active for each input.

FLOPs Formulas
Let:

d
i
n
d 
in
​
  = input dimension
d
o
u
t
d 
out
​
  = output dimension
n
n = total experts
k
k = active experts per input
Then:

Dense layer FLOPs:
FLOPs
dense
=
n
⋅
d
i
n
⋅
d
o
u
t
FLOPs 
dense
​
 =n⋅d 
in
​
 ⋅d 
out
​
 
MoE layer FLOPs:
FLOPs
moe
=
k
⋅
d
i
n
⋅
d
o
u
t
FLOPs 
moe
​
 =k⋅d 
in
​
 ⋅d 
out
​
 
Efficiency gain:
Savings
(
%
)
=
(
FLOPs
dense
−
FLOPs
moe
FLOPs
dense
)
⋅
100
Savings(%)=( 
FLOPs 
dense
​
 
FLOPs 
dense
​
 −FLOPs 
moe
​
 
​
 )⋅100
Example
Suppose:

n
=
1000
n=1000, 
k
=
2
k=2
d
i
n
=
d
o
u
t
=
512
d 
in
​
 =d 
out
​
 =512
Then:

MoE FLOPs = 
2
⋅
512
⋅
512
=
524
,
 ⁣
288
2⋅512⋅512=524,288
Full dense (all 1000 experts): 
1000
⋅
512
⋅
512
=
262
,
 ⁣
144
,
 ⁣
000
1000⋅512⋅512=262,144,000
Savings:
(
262
,
 ⁣
144
,
 ⁣
000
−
524
,
 ⁣
288
262
,
 ⁣
144
,
 ⁣
000
)
⋅
100
≈
99.8
%
( 
262,144,000
262,144,000−524,288
​
 )⋅100≈99.8%
This means the MoE layer uses just 0.2% of the computation compared to a full dense version â an enormous gain in efficiency.

Summary
By activating only a small number of experts per input, MoE layers reduce computation while maintaining high model capacity. This makes it feasible to train outrageously large models efficiently.

"""

def compute_efficiency(n_experts, k_active, d_in, d_out):
    """
    Calculate computational savings of MoE vs. dense layer.

    Args:
        n_experts: Total number of experts
        k_active: Number of active experts (sparsity)
        d_in: Input dimension
        d_out: Output dimension

    Returns:
        Percentage savings in FLOPs
    """
    F_dense = n_experts * d_in * d_out 
    F_moe = k_active * d_in * d_out 
    diff = F_dense - F_moe 
    sav = (diff / F_dense) * 100
    return sav
