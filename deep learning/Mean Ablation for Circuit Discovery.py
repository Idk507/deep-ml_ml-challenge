"""
Mean Ablation for Circuit Discovery
Medium
Deep Learning

Implement a mean ablation function for neural network interpretability. Mean ablation is a technique used to isolate neural circuits by replacing a node's activation with its mean value computed over a reference distribution. Given node activations, a binary mask indicating which nodes to ablate, and precomputed mean activations, return the ablated activations where masked nodes are replaced with their means while unmasked nodes retain their original values.

Example:
Input:
activations = [0.5, -0.3, 0.8, 0.2], mask = [1, 0, 1, 0], means = [0.1, 0.0, 0.2, -0.1]
Output:
[0.1, -0.3, 0.2, 0.2]
Reasoning:
Nodes at indices 0 and 2 have mask=1 so they are ablated (replaced with means 0.1 and 0.2). Nodes at indices 1 and 3 have mask=0 so they keep their original values (-0.3 and 0.2).

Learn About topic
Mean Ablation for Circuit Discovery
Mean ablation is a fundamental technique in mechanistic interpretability for isolating the minimal circuit responsible for a specific behavior in neural networks.

Core Idea
When analyzing neural networks, we want to identify which nodes (neurons, attention heads, residual channels) are necessary for a particular task. Mean ablation works by:

Computing the mean activation of each node over a reference distribution (typically the pretraining data)
"Ablating" nodes by replacing their activation with this mean value
Observing how task performance changes
Mathematical Definition
For a node with activation 
x
i
x 
i
​
  and precomputed mean 
μ
i
μ 
i
​
 , mean ablation with mask 
m
i
∈
{
0
,
1
}
m 
i
​
 ∈{0,1} produces: 
x
i
ablated
=
m
i
⋅
μ
i
+
(
1
−
m
i
)
⋅
x
i
x 
i
ablated
​
 =m 
i
​
 ⋅μ 
i
​
 +(1−m 
i
​
 )⋅x 
i
​
 

where:

m
i
=
1
m 
i
​
 =1: node is ablated (replaced with mean)
m
i
=
0
m 
i
​
 =0: node is preserved (keeps original activation)
Why Mean Ablation?
Mean ablation has several advantages over zero ablation:

Preserves distribution: The mean is a natural "neutral" value that doesn't push the network far off its typical activation manifold

Removes information: By fixing a node to its mean, we remove all task-specific information it might carry

Validates circuits: If ablating all nodes except a proposed circuit preserves task performance, the circuit is sufficient. If ablating only the circuit destroys performance, it is necessary.

Application to Sparse Circuits
In weight-sparse transformer interpretability, mean ablation is used to prune models down to minimal circuits. The process involves learning binary masks that minimize both task loss and circuit size, where masked-out nodes are mean-ablated.

This technique enables researchers to isolate circuits with as few as 10-20 nodes that fully explain specific model behaviors.

"""
import numpy as np

def mean_ablate(activations: np.ndarray, mask: np.ndarray, means: np.ndarray) -> np.ndarray:
    """
    Apply mean ablation to node activations.
    
    Args:
        activations: Original node activations, shape (n,) or (batch, n)
        mask: Binary mask where 1 = ablate (replace with mean), 0 = keep original
        means: Precomputed mean activations for each node, shape (n,)
    
    Returns:
        Ablated activations with same shape as input
    """
    # Apply the ablation formula: x_ablated = mask * mean + (1 - mask) * x
    # This works for both 1D (n,) and 2D (batch, n) cases due to broadcasting
    return mask * means + (1 - mask) * activations
