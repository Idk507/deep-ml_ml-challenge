"""
Implement a function entropy_split_selection that finds the best feature and threshold for splitting a dataset based on information gain using entropy. Given a feature matrix X of shape (n_samples, n_features) and corresponding labels y of shape (n_samples,), the function should evaluate all possible binary splits for each feature and return the split that maximizes information gain.

For continuous features, consider split points as midpoints between consecutive unique values. For each potential split, calculate the information gain as the reduction in entropy from the parent node to the weighted average entropy of the child nodes.

The function should return a tuple containing:

best_feature_index: Index of the feature that provides the best split
best_threshold: The threshold value for the best split
best_info_gain: The information gain achieved by the best split
Handle edge cases such as pure nodes (all samples have the same label) and features with only one unique value.

Example:
Input:
X = np.array([[1], [2], [3], [4]]), y = np.array([0, 0, 1, 1])
Output:
(0, 2.5, 1.0)
Reasoning:
Parent entropy: H = -0.5log2(0.5) - 0.5log2(0.5) = 1.0. Split at threshold 2.5: Left child has labels [0, 0] with entropy 0, right child has labels [1, 1] with entropy 0. Weighted child entropy = 0. Information gain = 1.0 - 0 = 1.0. This is a perfect split that completely separates the classes.

Learn About topic
Entropy-based Split Selection for Decision Trees
Entropy
Entropy is a measure of impurity or disorder in a dataset. For a set 
S
S with 
K
K classes, the entropy is defined as:

H
(
S
)
=
−
∑
k
=
1
K
p
k
log
⁡
2
(
p
k
)
H(S)=− 
k=1
∑
K
​
 p 
k
​
 log 
2
​
 (p 
k
​
 )
where 
p
k
p 
k
​
  is the proportion of samples belonging to class 
k
k.

Key properties:

H
(
S
)
=
0
H(S)=0 when all samples belong to one class (pure node)
H
(
S
)
H(S) is maximized when samples are evenly distributed across classes
For binary classification: 
H
m
a
x
=
1
H 
max
​
 =1 bit
For 
K
K classes: 
H
m
a
x
=
log
⁡
2
(
K
)
H 
max
​
 =log 
2
​
 (K) bits
Information Gain
Information gain measures the reduction in entropy achieved by splitting a dataset on a particular feature. For a split that divides set 
S
S into subsets 
S
l
e
f
t
S 
left
​
  and 
S
r
i
g
h
t
S 
right
​
 :

I
G
(
S
,
split
)
=
H
(
S
)
−
(
∣
S
l
e
f
t
∣
∣
S
∣
H
(
S
l
e
f
t
)
+
∣
S
r
i
g
h
t
∣
∣
S
∣
H
(
S
r
i
g
h
t
)
)
IG(S,split)=H(S)−( 
∣S∣
∣S 
left
​
 ∣
​
 H(S 
left
​
 )+ 
∣S∣
∣S 
right
​
 ∣
​
 H(S 
right
​
 ))
The weighted sum on the right is called the weighted child entropy.

Finding the Best Split
For continuous features:

Sort unique feature values
Consider midpoints between consecutive values as potential thresholds
For each threshold:
Split samples: left (
x
≤
t
h
r
e
s
h
o
l
d
x≤threshold) and right (
x
>
t
h
r
e
s
h
o
l
d
x>threshold)
Calculate information gain
Select the feature and threshold with maximum information gain
Example Calculation
Given labels 
y
=
[
0
,
0
,
1
,
1
]
y=[0,0,1,1]:

Parent entropy:

H
(
S
)
=
−
2
4
log
⁡
2
(
2
4
)
−
2
4
log
⁡
2
(
2
4
)
=
1.0
H(S)=− 
4
2
​
 log 
2
​
 ( 
4
2
​
 )− 
4
2
​
 log 
2
​
 ( 
4
2
​
 )=1.0
After split at threshold 2.5:

Left: 
[
0
,
0
]
[0,0] with 
H
=
0
H=0 (pure)
Right: 
[
1
,
1
]
[1,1] with 
H
=
0
H=0 (pure)
I
G
=
1.0
−
(
2
4
⋅
0
+
2
4
⋅
0
)
=
1.0
IG=1.0−( 
4
2
​
 ⋅0+ 
4
2
​
 ⋅0)=1.0
Comparison with Gini Impurity
Both entropy and Gini impurity measure node purity. Entropy tends to produce slightly more balanced trees, while Gini is computationally faster (no logarithm). In practice, they often produce similar results.


"""
import numpy as np

def entropy_split_selection(X: np.ndarray, y: np.ndarray) -> tuple:
    """
    Find the best feature and threshold for splitting based on information gain.
    
    Args:
        X: Feature matrix of shape (n_samples, n_features)
        y: Labels of shape (n_samples,)
    
    Returns:
        Tuple of (best_feature_index, best_threshold, best_info_gain)
    """
    def entropy(labels):
        """Calculate entropy for a set of labels."""
        n = len(labels)
        if n == 0:
            return 0.0
        
        # Count occurrences of each class
        _, counts = np.unique(labels, return_counts=True)
        
        # Compute probabilities
        probs = counts / n
        
        # Entropy: -sum(p * log2(p)), skip zero probabilities
        return -np.sum(probs * np.log2(probs + 1e-12))
    
    def information_gain(y_parent, y_left, y_right):
        """Calculate information gain of a split."""
        n = len(y_parent)
        n_left = len(y_left)
        n_right = len(y_right)
        
        # Parent entropy
        H_parent = entropy(y_parent)
        
        # Weighted child entropy
        weighted_child_entropy = (n_left / n) * entropy(y_left) + \
                                  (n_right / n) * entropy(y_right)
        
        # Information gain
        return H_parent - weighted_child_entropy
    
    n_samples, n_features = X.shape
    
    # Initialize best split tracking
    best_feature_idx = None
    best_threshold = None
    best_info_gain = -np.inf
    
    # Early exit: pure node (all labels are the same)
    if len(np.unique(y)) == 1:
        return (0, X[0, 0], 0.0)
    
    # Evaluate each feature
    for feature_idx in range(n_features):
        feature_values = X[:, feature_idx]
        
        # Get sorted unique values
        unique_values = np.unique(feature_values)
        
        # Skip features with only one unique value (no split possible)
        if len(unique_values) < 2:
            continue
        
        # Generate candidate thresholds: midpoints between consecutive unique values
        thresholds = (unique_values[:-1] + unique_values[1:]) / 2
        
        # Evaluate each threshold
        for threshold in thresholds:
            # Split labels into left (≤ threshold) and right (> threshold)
            left_mask = feature_values <= threshold
            right_mask = ~left_mask
            
            y_left = y[left_mask]
            y_right = y[right_mask]
            
            # Skip degenerate splits where one side is empty
            if len(y_left) == 0 or len(y_right) == 0:
                continue
            
            # Calculate information gain
            ig = information_gain(y, y_left, y_right)
            
            # Update best split if this is better
            if ig > best_info_gain:
                best_info_gain = ig
                best_feature_idx = feature_idx
                best_threshold = threshold
    
    # Fallback if no valid split was found
    if best_feature_idx is None:
        return (0, X[0, 0], 0.0)
    
    return (best_feature_idx, best_threshold, round(best_info_gain, 10))

