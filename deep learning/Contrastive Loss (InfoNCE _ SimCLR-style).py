"""

Implement the NT-Xent (Normalized Temperature-scaled Cross-Entropy) contrastive loss function, commonly known as the SimCLR contrastive loss or InfoNCE loss.

This loss is the backbone of self-supervised contrastive learning frameworks. Given a batch of 2N embedding vectors where each consecutive pair (2i, 2i+1) represents two augmented views of the same input sample, the goal is to push representations of the same sample closer together while pushing representations of different samples apart.

Your function receives:

embeddings: a numpy array of shape (2N, d) where N is the number of original samples and d is the embedding dimension. Embeddings at indices (0,1), (2,3), (4,5), ... form positive pairs.
temperature: a positive float scaling parameter (tau).
The function should:

Compute pairwise cosine similarities between all L2-normalized embeddings.
For each of the 2N anchors, treat its paired view as the positive and all other 2N-2 samples (excluding itself) as negatives.
Compute the contrastive loss for each anchor and return the mean loss over all 2N anchors.
Use the log-sum-exp trick for numerical stability when computing the log of the sum of exponentials.

Example:
Input:
embeddings = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]), temperature = 1.0
Output:
0.5514
Reasoning:
There are 2 positive pairs: (index 0, index 1) both [1,0] and (index 2, index 3) both [0,1]. After L2 normalization (already unit vectors), the cosine similarity matrix is: sim[0,1]=1, sim[0,2]=0, sim[0,3]=0 (and symmetrically for others). For anchor i=0 with positive j=1: the numerator uses sim(0,1)/tau = 1.0. The denominator sums exp of similarities with all except self: exp(1) + exp(0) + exp(0) = e + 2 = 4.7183. Loss for anchor 0: -1 + ln(4.7183) = -1 + 1.5514 = 0.5514. By the symmetry of the embeddings, every anchor produces the same loss. The mean loss is 0.5514.

Learn About topic
Contrastive Loss: InfoNCE / NT-Xent
Motivation
In self-supervised learning, we want to learn representations without labels. Contrastive learning achieves this by learning to distinguish between similar (positive) and dissimilar (negative) pairs. The NT-Xent (Normalized Temperature-scaled Cross-Entropy) loss, used in SimCLR, is one of the most successful contrastive objectives.

Setup
Given a minibatch of 
N
N samples, we create two augmented views of each, yielding 
2
N
2N embedding vectors 
{
z
1
,
z
2
,
…
,
z
2
N
}
{z 
1
​
 ,z 
2
​
 ,…,z 
2N
​
 }. Pairs 
(
z
2
k
,
z
2
k
+
1
)
(z 
2k
​
 ,z 
2k+1
​
 ) for 
k
=
0
,
1
,
…
,
N
−
1
k=0,1,…,N−1 are positive pairs (derived from the same input).

Cosine Similarity
First, all embeddings are L2-normalized. The similarity between two vectors is the cosine similarity:

sim
(
u
,
v
)
=
u
⋅
v
∥
u
∥
 
∥
v
∥
sim(u,v)= 
∥u∥∥v∥
u⋅v
​
 

After normalization, this reduces to the dot product.

The NT-Xent Loss
For an anchor 
i
i with its positive pair 
j
j, the loss is:

ℓ
(
i
,
j
)
=
−
sim
(
z
i
,
z
j
)
τ
+
log
⁡
∑
k
=
1
k
≠
i
2
N
exp
⁡
(
sim
(
z
i
,
z
k
)
τ
)
ℓ(i,j)=− 
τ
sim(z 
i
​
 ,z 
j
​
 )
​
 +log∑ 
k=1
k

=i
​
 
2N
​
 exp( 
τ
sim(z 
i
​
 ,z 
k
​
 )
​
 )

where 
τ
>
0
τ>0 is the temperature parameter. The total loss is the mean over all 
2
N
2N anchors:

L
=
1
2
N
∑
i
=
1
2
N
ℓ
(
i
,
j
(
i
)
)
L= 
2N
1
​
 ∑ 
i=1
2N
​
 ℓ(i,j(i))

where 
j
(
i
)
j(i) is the index of the positive partner of anchor 
i
i.

Understanding Temperature
Small 
τ
τ: Sharpens the distribution, making the model more sensitive to hard negatives. The loss focuses on separating the most confusing pairs.
Large 
τ
τ: Smooths the distribution, treating all negatives more equally.
Typical values range from 0.05 to 1.0, with 
τ
=
0.5
τ=0.5 being common in SimCLR.

Connection to InfoNCE
The NT-Xent loss is a specific instance of the InfoNCE (Noise-Contrastive Estimation) bound from Contrastive Predictive Coding (CPC). InfoNCE provides a lower bound on the mutual information between the two views:

I
(
z
i
;
z
j
)
≥
log
⁡
(
2
N
−
1
)
−
L
InfoNCE
I(z 
i
​
 ;z 
j
​
 )≥log(2N−1)−L 
InfoNCE
​
 

Minimizing the loss maximizes a lower bound on mutual information.

Numerical Stability: Log-Sum-Exp Trick
When computing 
log
⁡
∑
k
exp
⁡
(
a
k
)
log∑ 
k
​
 exp(a 
k
​
 ), direct computation can cause numerical overflow. The log-sum-exp trick uses:

log
⁡
∑
k
exp
⁡
(
a
k
)
=
m
+
log
⁡
∑
k
exp
⁡
(
a
k
−
m
)
log∑ 
k
​
 exp(a 
k
​
 )=m+log∑ 
k
​
 exp(a 
k
​
 −m)

where 
m
=
max
⁡
k
a
k
m=max 
k
​
 a 
k
​
 . This ensures the largest exponent is 
exp
⁡
(
0
)
=
1
exp(0)=1, preventing overflow.

Why Contrastive Learning Works
By pulling positive pairs together and pushing negatives apart in embedding space, the model learns features that capture semantic similarity. The key insight is that augmented views of the same input should share high-level information (content) while differing in low-level details (style, noise). The learned representations transfer well to downstream tasks even without any labeled data during pretraining.
"""
import numpy as np

def contrastive_loss(embeddings: np.ndarray, temperature: float) -> float:
    """
    Compute the NT-Xent (SimCLR-style) contrastive loss.
    
    Args:
        embeddings: Array of shape (2N, d) where consecutive pairs
                    (2i, 2i+1) are positive pairs.
        temperature: Temperature scaling parameter (tau > 0).
    
    Returns:
        The mean contrastive loss as a float.
    """
    # L2-normalize embeddings
    embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
    
    # Compute pairwise cosine similarities (dot products after normalization)
    # similarity[i, j] = embeddings[i] · embeddings[j]
    similarity = np.matmul(embeddings, embeddings.T)
    
    # Scale by temperature
    similarity = similarity / temperature
    
    # Number of samples (2N)
    num_samples = embeddings.shape[0]
    
    # Create mask for positive pairs
    # For indices (0,1), (2,3), (4,5), etc., they are positive pairs
    positive_mask = np.zeros((num_samples, num_samples), dtype=bool)
    for i in range(num_samples):
        if i % 2 == 0:
            positive_mask[i, i + 1] = True
        else:
            positive_mask[i, i - 1] = True
    
    # Create mask for valid negatives (all except self)
    # We exclude the anchor itself from the denominator
    negatives_mask = np.ones((num_samples, num_samples), dtype=bool)
    np.fill_diagonal(negatives_mask, False)
    
    total_loss = 0.0
    
    # Compute loss for each anchor
    for i in range(num_samples):
        # Find the positive pair for anchor i
        positive_idx = np.where(positive_mask[i])[0][0]
        
        # Numerator: similarity with positive pair
        numerator = similarity[i, positive_idx]
        
        # Denominator: sum of exp(similarities) with all samples except self
        # This includes the positive and all negatives
        # Use log-sum-exp trick for numerical stability
        logits = similarity[i, negatives_mask[i]]
        max_logit = np.max(logits)
        denominator_log = max_logit + np.log(np.sum(np.exp(logits - max_logit)))
        
        # Loss for this anchor: -numerator + log(denominator)
        loss_i = -numerator + denominator_log
        total_loss += loss_i
    
    # Return mean loss over all anchors
    return float(total_loss / num_samples)


