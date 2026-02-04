"""
Implement a function to compute the hinge loss, which is the standard loss function used in Support Vector Machine (SVM) classifiers for maximum-margin classification.

Given an array of true labels (where each label is either -1 or +1) and an array of predicted scores (raw SVM output scores, not probabilities), compute the average hinge loss across all samples.

The hinge loss penalizes predictions that are on the wrong side of the decision boundary or within the margin. Correct predictions with sufficient margin (score * label >= 1) incur zero loss.

Write a function hinge_loss(y_true, y_pred) that:

Takes y_true: numpy array of true labels (-1 or +1)
Takes y_pred: numpy array of raw prediction scores
Returns the average hinge loss as a float, rounded to 4 decimal places
Example:
Input:
y_true = np.array([1, -1, 1, -1]), y_pred = np.array([0.5, -0.5, -0.2, 0.3])
Output:
0.875
Reasoning:
For each sample, compute max(0, 1 - y_true * y_pred): Sample 1: max(0, 1 - 10.5) = max(0, 0.5) = 0.5. Sample 2: max(0, 1 - (-1)(-0.5)) = max(0, 0.5) = 0.5. Sample 3: max(0, 1 - 1*(-0.2)) = max(0, 1.2) = 1.2. Sample 4: max(0, 1 - (-1)*(0.3)) = max(0, 1.3) = 1.3. Average = (0.5 + 0.5 + 1.2 + 1.3) / 4 = 3.5 / 4 = 0.875

Learn About topic
Understanding Hinge Loss for SVMs
Hinge loss is the standard loss function used to train Support Vector Machine (SVM) classifiers. It is designed to maximize the margin between classes while penalizing misclassifications.

Mathematical Definition
For a single sample with true label 
y
i
∈
{
−
1
,
+
1
}
y 
i
​
 ∈{−1,+1} and predicted score 
y
^
i
y
^
​
  
i
​
 , the hinge loss is defined as:

L
i
=
max
⁡
(
0
,
1
−
y
i
⋅
y
^
i
)
L 
i
​
 =max(0,1−y 
i
​
 ⋅ 
y
^
​
  
i
​
 )

For a batch of 
n
n samples, the average hinge loss is:

L
=
1
n
∑
i
=
1
n
max
⁡
(
0
,
1
−
y
i
⋅
y
^
i
)
L= 
n
1
​
 ∑ 
i=1
n
​
 max(0,1−y 
i
​
 ⋅ 
y
^
​
  
i
​
 )

Intuition
The hinge loss has three important regions:

Correct with sufficient margin (
y
i
⋅
y
^
i
≥
1
y 
i
​
 ⋅ 
y
^
​
  
i
​
 ≥1): Loss is 0. The prediction is correct and confident.

Correct but within margin (
0
<
y
i
⋅
y
^
i
<
1
0<y 
i
​
 ⋅ 
y
^
​
  
i
​
 <1): Loss is 
(
1
−
y
i
⋅
y
^
i
)
(1−y 
i
​
 ⋅ 
y
^
​
  
i
​
 ). The prediction is correct but not confident enough.

Incorrect (
y
i
⋅
y
^
i
≤
0
y 
i
​
 ⋅ 
y
^
​
  
i
​
 ≤0): Loss is at least 1. The prediction is wrong.

The Margin Concept
In SVMs, the goal is to find a hyperplane that maximizes the margin between classes. Points exactly on the margin satisfy 
y
i
⋅
y
^
i
=
1
y 
i
​
 ⋅ 
y
^
​
  
i
​
 =1. The hinge loss:

Is zero for points outside the margin (correctly classified with high confidence)
Increases linearly for points inside the margin or misclassified
Example Calculation
Consider 
y
t
r
u
e
=
[
1
,
−
1
]
y 
true
​
 =[1,−1] and 
y
p
r
e
d
=
[
0.5
,
0.5
]
y 
pred
​
 =[0.5,0.5]:

Sample 1: 
max
⁡
(
0
,
1
−
1
×
0.5
)
=
max
⁡
(
0
,
0.5
)
=
0.5
max(0,1−1×0.5)=max(0,0.5)=0.5
Sample 2: 
max
⁡
(
0
,
1
−
(
−
1
)
×
0.5
)
=
max
⁡
(
0
,
1.5
)
=
1.5
max(0,1−(−1)×0.5)=max(0,1.5)=1.5
Average loss = 
(
0.5
+
1.5
)
/
2
=
1.0
(0.5+1.5)/2=1.0

Properties
Convex: Hinge loss is convex, making optimization tractable
Sparse gradients: Zero gradient for well-classified points, leading to sparse solutions
Non-differentiable at 1: The hinge function has a kink at 
y
i
⋅
y
^
i
=
1
y 
i
​
 ⋅ 
y
^
​
  
i
​
 =1, requiring subgradient methods

 """

import numpy as np

def hinge_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute the average hinge loss for SVM classification.
    
    Args:
        y_true: Array of true labels (-1 or +1)
        y_pred: Array of predicted scores (raw SVM scores)
    
    Returns:
        Average hinge loss rounded to 4 decimal places
    """
    # Convert inputs to numpy arrays if needed
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Compute hinge loss for each sample: max(0, 1 - y_true * y_pred)
    # Element-wise multiplication: y_true * y_pred gives the margin
    margins = y_true * y_pred
    
    # Hinge loss: max(0, 1 - margin)
    losses = np.maximum(0, 1 - margins)
    
    # Compute average loss
    avg_loss = np.mean(losses)
    
    # Round to 4 decimal places
    return round(float(avg_loss), 4)
