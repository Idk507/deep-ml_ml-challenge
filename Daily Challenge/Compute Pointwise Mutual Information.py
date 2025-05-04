"""
Implement a function to compute the Pointwise Mutual Information (PMI) given the joint occurrence count of two events, their individual counts, and the total number of samples. PMI measures how much the actual joint occurrence of events differs from what we would expect by chance.

Example:
Input:
compute_pmi(50, 200, 300, 1000)
Output:
-0.263
Reasoning:
The PMI calculation compares the actual joint probability (50/1000 = 0.05) to the product of the individual probabilities (200/1000 * 300/1000 = 0.06). Thus, PMI = log₂(0.05 / (0.2 * 0.3)) ≈ -0.263, indicating the events co-occur slightly less than expected by chance.

Pointwise Mutual Information (PMI)
Pointwise Mutual Information (PMI) is a statistical measure used in information theory and Natural Language Processing (NLP) to quantify the association between two events. It measures how much the actual joint occurrence of two events differs from what would be expected if they were independent. PMI is commonly used for identifying word associations, feature selection in text classification, and calculating document similarity.

Implementation
Collect Count Data for events 
x
x, 
y
y, and their joint occurrence 
(
x
,
y
)
(x,y).

Calculate Individual Probabilities:

P
(
x
)
=
Count
(
x
)
Total Count
P(x)= 
Total Count
Count(x)
​
 
P
(
y
)
=
Count
(
y
)
Total Count
P(y)= 
Total Count
Count(y)
​
 
Calculate Joint Probability:

P
(
x
,
y
)
=
Count
(
x
,
y
)
Total Count
P(x,y)= 
Total Count
Count(x,y)
​
 
Calculate PMI:

PMI
(
x
,
y
)
=
log
⁡
2
(
P
(
x
,
y
)
P
(
x
)
⋅
P
(
y
)
)
PMI(x,y)=log 
2
​
 ( 
P(x)⋅P(y)
P(x,y)
​
 )
Interpretation of PMI Values
Positive PMI: Events co-occur more frequently than expected by chance.
Zero PMI: Events are statistically independent.
Negative PMI: Events co-occur less frequently than expected by chance.
Undefined PMI: Occurs when 
P
(
x
,
y
)
=
0
P(x,y)=0 (the events never co-occur).
Variants of PMI
1. Normalized PMI (NPMI)
NPMI scales PMI to a range of [-1, 1] to account for dataset size variations:

NPMI
(
x
,
y
)
=
PMI
(
x
,
y
)
−
log
⁡
2
P
(
x
,
y
)
NPMI(x,y)= 
−log 
2
​
 P(x,y)
PMI(x,y)
​
 
2. Positive PMI (PPMI)
PPMI sets negative PMI scores to zero, often used in word embeddings:

PPMI
(
x
,
y
)
=
max
⁡
(
PMI
(
x
,
y
)
,
 
0
)
PPMI(x,y)=max(PMI(x,y),0)

"""
import numpy as np

def compute_pmi(joint_counts, total_counts_x, total_counts_y, total_samples):

    if not all(isinstance(x, int) and x >= 0 for x in [joint_counts, total_counts_x, total_counts_y, total_samples]):
        raise ValueError("All inputs must be non-negative integers.")

    if total_samples == 0:
        raise ValueError("Total samples cannot be zero.")

    if joint_counts > min(total_counts_x, total_counts_y):
        raise ValueError("Joint counts cannot exceed individual counts.")

    p_x = total_counts_x / total_samples
    p_y = total_counts_y / total_samples
    p_xy = joint_counts / total_samples

    if p_xy == 0:
        return float('-inf')

    pmi = np.log2(p_xy / (p_x * p_y))

    return round(pmi, 3)
