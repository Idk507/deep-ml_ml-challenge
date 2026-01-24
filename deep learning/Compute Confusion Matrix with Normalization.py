"""
Implement a function to compute a confusion matrix for multi-class classification with optional normalization. The function should support three normalization modes: by true labels (row-wise), by predicted labels (column-wise), and by all samples (global). When normalization is enabled, round the outputs to a configurable number of decimal places.

Example:
Input:
compute_confusion_matrix([0,1,2,2], [0,2,1,2], 3, normalize='true', round_decimals=4)
Output:
[[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.5, 0.5]]
Reasoning:
Counts: [[1,0,0],[0,0,1],[0,1,1]]. Row-wise normalization divides each row by its sum; rows with sums 1 stay the same, the last row [0,1,1]/2 -> [0.0, 0.5, 0.5].

Learn About topic
Confusion Matrix & Normalization
A confusion matrix for a 
K
K-class classifier is a 
K
×
K
K×K table 
C
C where 
C
i
j
=
∑
n
=
1
N
1
[
y
(
n
)
=
i
∧
y
^
(
n
)
=
j
]
C 
ij
​
 =∑ 
n=1
N
​
 1[y 
(n)
 =i∧ 
y
^
​
  
(n)
 =j] Rows index the true class 
i
i and columns index the predicted class 
j
j.

Normalization schemes
None (counts): Return raw counts 
C
i
j
C 
ij
​
 .
By true labels (row-wise): 
C
i
j
(
true
)
=
{
C
i
j
∑
j
′
C
i
j
′
if 
∑
j
′
C
i
j
′
>
0
0
otherwise
C 
ij
(true)
​
 = 
⎩
⎨
⎧
​
  
∑ 
j 
′
 
​
 C 
ij 
′
 
​
 
C 
ij
​
 
​
 
0
​
  
if ∑ 
j 
′
 
​
 C 
ij 
′
 
​
 >0
otherwise
​
  Interpreted as 
P
(
Y
^
=
j
∣
Y
=
i
)
P( 
Y
^
 =j∣Y=i).
By predicted labels (column-wise): 
C
i
j
(
pred
)
=
{
C
i
j
∑
i
′
C
i
′
j
if 
∑
i
′
C
i
′
j
>
0
0
otherwise
C 
ij
(pred)
​
 = 
⎩
⎨
⎧
​
  
∑ 
i 
′
 
​
 C 
i 
′
 j
​
 
C 
ij
​
 
​
 
0
​
  
if ∑ 
i 
′
 
​
 C 
i 
′
 j
​
 >0
otherwise
​
  Interpreted as 
P
(
Y
=
i
∣
Y
^
=
j
)
P(Y=i∣ 
Y
^
 =j).
By all samples (global): 
C
i
j
(
all
)
=
C
i
j
∑
i
′
,
j
′
C
i
′
j
′
=
C
i
j
N
C 
ij
(all)
​
 = 
∑ 
i 
′
 ,j 
′
 
​
 C 
i 
′
 j 
′
 
​
 
C 
ij
​
 
​
 = 
N
C 
ij
​
 
​
  Interpreted as joint probability 
P
(
Y
=
i
,
Y
^
=
j
)
P(Y=i, 
Y
^
 =j).
Notes
Use 
K
K (number of classes) to size the matrix even if some classes are absent.
When normalizing, round to a specified number of decimals (e.g., 4). Avoid division by zero by leaving rows/columns as zeros when their sums are zero.
"""
import numpy as np

def compute_confusion_matrix(y_true, y_pred, num_classes, normalize=None, round_decimals=4):
    """
    Compute a KxK confusion matrix with optional normalization.

    Args:
        y_true: Iterable of true labels in [0, K-1]
        y_pred: Iterable of predicted labels in [0, K-1]
        num_classes: K, number of classes
        normalize: None | 'true' | 'pred' | 'all'
        round_decimals: decimals to round when normalization is applied

    Returns:
        list[list[int|float]] confusion matrix
    """
    # Convert to numpy arrays for easier manipulation
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Initialize confusion matrix with zeros
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
    
    # Populate the confusion matrix
    # C[i,j] = count where true label is i and predicted label is j
    for true_label, pred_label in zip(y_true, y_pred):
        confusion_matrix[true_label, pred_label] += 1
    
    # Apply normalization if requested
    if normalize is not None:
        # Convert to float for normalization
        confusion_matrix = confusion_matrix.astype(float)
        
        if normalize == 'true':
            # Row-wise normalization: divide each row by its sum
            # P(Y_pred=j | Y_true=i)
            row_sums = confusion_matrix.sum(axis=1, keepdims=True)
            # Avoid division by zero: only divide where row_sum > 0
            with np.errstate(divide='ignore', invalid='ignore'):
                confusion_matrix = np.where(
                    row_sums > 0,
                    confusion_matrix / row_sums,
                    0.0
                )
        
        elif normalize == 'pred':
            # Column-wise normalization: divide each column by its sum
            # P(Y_true=i | Y_pred=j)
            col_sums = confusion_matrix.sum(axis=0, keepdims=True)
            # Avoid division by zero: only divide where col_sum > 0
            with np.errstate(divide='ignore', invalid='ignore'):
                confusion_matrix = np.where(
                    col_sums > 0,
                    confusion_matrix / col_sums,
                    0.0
                )
        
        elif normalize == 'all':
            # Global normalization: divide by total number of samples
            # P(Y_true=i, Y_pred=j)
            total = confusion_matrix.sum()
            if total > 0:
                confusion_matrix = confusion_matrix / total
        
        # Round to specified decimal places
        confusion_matrix = np.round(confusion_matrix, decimals=round_decimals)
    
    # Convert to list of lists
    return confusion_matrix.tolist()
