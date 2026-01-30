"""
Implement the Focal Loss function, which is designed to address class imbalance in classification tasks by down-weighting easy examples and focusing on hard misclassified examples. Given ground truth labels (as class indices), predicted probabilities, a focusing parameter gamma, and optional class weights alpha, compute the average focal loss across all samples.

The focal loss modifies the standard cross-entropy loss by adding a modulating factor that reduces the loss contribution from well-classified examples, allowing the model to focus learning on hard examples.

Your function should:

Handle both numpy arrays and Python lists as inputs
Apply clipping to predictions to avoid numerical issues with log(0)
Support an optional alpha parameter for class weighting
Return the average focal loss as a float
Example:
Input:
y_true = [0, 1, 2], y_pred = [[0.9, 0.05, 0.05], [0.1, 0.8, 0.1], [0.1, 0.2, 0.7]], gamma = 2.0
Output:
0.014
Reasoning:
For each sample, we first get the predicted probability for the true class: pt = [0.9, 0.8, 0.7]. Then we compute the focal weight: (1-pt)^gamma = [0.01, 0.04, 0.09]. The cross-entropy for each sample is: -log(pt) = [0.1054, 0.2231, 0.3567]. The focal loss for each sample is: focal_weight * ce = [0.00105, 0.00893, 0.0321]. The average focal loss is (0.00105 + 0.00893 + 0.0321) / 3 = 0.014.

Learn About topic
Understanding Focal Loss
Focal Loss was introduced in the paper "Focal Loss for Dense Object Detection" (Lin et al., 2017) to address the class imbalance problem in object detection, where background examples vastly outnumber foreground objects.

The Problem with Standard Cross-Entropy
Standard cross-entropy loss for a single sample is:

C
E
(
p
t
)
=
−
log
⁡
(
p
t
)
CE(p 
t
​
 )=−log(p 
t
​
 )
where 
p
t
p 
t
​
  is the predicted probability for the true class. The issue is that even easily classified examples (high 
p
t
p 
t
​
 ) contribute to the loss, which can overwhelm the gradient from hard examples.

Focal Loss Definition
Focal Loss adds a modulating factor 
(
1
−
p
t
)
γ
(1−p 
t
​
 ) 
γ
  to the cross-entropy:

F
L
(
p
t
)
=
−
(
1
−
p
t
)
γ
log
⁡
(
p
t
)
FL(p 
t
​
 )=−(1−p 
t
​
 ) 
γ
 log(p 
t
​
 )
where:

p
t
p 
t
​
  is the model's estimated probability for the true class
γ
≥
0
γ≥0 is the focusing parameter
How the Focusing Parameter Works
When 
γ
=
0
γ=0, Focal Loss equals standard Cross-Entropy
As 
γ
γ increases, the effect of the modulating factor increases
For well-classified examples where 
p
t
p 
t
​
  is large (e.g., 0.9), 
(
1
−
p
t
)
γ
(1−p 
t
​
 ) 
γ
  becomes very small, reducing their contribution
For misclassified examples where 
p
t
p 
t
​
  is small (e.g., 0.1), the modulating factor is near 1, maintaining their full contribution
Adding Class Weights
To handle class imbalance further, we can add a weighting factor 
α
t
α 
t
​
 :

F
L
(
p
t
)
=
−
α
t
(
1
−
p
t
)
γ
log
⁡
(
p
t
)
FL(p 
t
​
 )=−α 
t
​
 (1−p 
t
​
 ) 
γ
 log(p 
t
​
 )
where 
α
t
α 
t
​
  is the weight for the true class.

Example Calculation
Consider a sample with true class 0 and predictions 
[
0.9
,
0.1
]
[0.9,0.1]:

p
t
=
0.9
p 
t
​
 =0.9 (well-classified)
With 
γ
=
2
γ=2: 
(
1
−
0.9
)
2
=
0.01
(1−0.9) 
2
 =0.01
Cross-entropy: 
−
log
⁡
(
0.9
)
≈
0.105
−log(0.9)≈0.105
Focal Loss: 
0.01
×
0.105
=
0.00105
0.01×0.105=0.00105
Compare to a misclassified example with 
p
t
=
0.2
p 
t
​
 =0.2:

With 
γ
=
2
γ=2: 
(
1
−
0.2
)
2
=
0.64
(1−0.2) 
2
 =0.64
Cross-entropy: 
−
log
⁡
(
0.2
)
≈
1.609
−log(0.2)≈1.609
Focal Loss: 
0.64
×
1.609
=
1.03
0.64×1.609=1.03
The focal loss for the hard example is nearly 1000x larger than for the easy example, allowing the model to focus on learning from hard cases.

"""

import numpy as np

def focal_loss(y_true, y_pred, gamma=2.0, alpha=None):
    """
    Compute Focal Loss for multi-class classification.
    
    Args:
        y_true: Ground truth labels as class indices (list or 1D array)
        y_pred: Predicted probabilities (2D array, shape: [n_samples, n_classes])
        gamma: Focusing parameter (default: 2.0)
        alpha: Class weights (optional, list or 1D array of length n_classes)
    
    Returns:
        float: Average focal loss
    """
    # Convert inputs to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Clip predictions to avoid log(0) and numerical instability
    epsilon = 1e-7
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    
    # Get number of samples and classes
    n_samples = y_pred.shape[0]
    
    # Extract the predicted probability for the true class for each sample
    # p_t[i] = y_pred[i, y_true[i]]
    p_t = y_pred[np.arange(n_samples), y_true]
    
    # Compute the focal weight: (1 - p_t)^gamma
    focal_weight = np.power(1 - p_t, gamma)
    
    # Compute cross-entropy: -log(p_t)
    ce_loss = -np.log(p_t)
    
    # Compute focal loss for each sample: focal_weight * ce_loss
    focal_loss_per_sample = focal_weight * ce_loss
    
    # Apply class weights (alpha) if provided
    if alpha is not None:
        alpha = np.array(alpha)
        # Get the alpha weight for each sample's true class
        alpha_t = alpha[y_true]
        focal_loss_per_sample = alpha_t * focal_loss_per_sample
    
    # Return the average focal loss
    return float(np.mean(focal_loss_per_sample))
