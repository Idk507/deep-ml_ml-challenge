"""
Deep in the Crystal Cave, the enigmatic Pattern Weaver creates stunning sequences by uncovering the intricate relationships between crystals. Each crystal is marked by a unique numeric value, and the Weaver emphasizes that the true power of any crystal depends on how it interacts with all others. You have discovered N crystals, each with a specific value, and your task is to reveal their enhanced patterns by analyzing these relationships using self-attention. Given a sequence of crystals and their values, your task is to implement a simplified self-attention mechanism. For each crystal, calculate its relationship with every other crystal, compute the attention scores using the softmax function, and derive the final weighted pattern for each crystal. This Problem was made with the help of GroundZero AI

Example:
Input:
number of crystals: 5
values: 4 2 7 1 9
 dimension: 1
Output:
[8.9993, 8.9638, 9.0, 8.7259, 9.0]
Reasoning:
The self-attention mechanism calculates relationships (attention scores) for each crystal using the given formula. These scores are converted to probabilities using the softmax function, and the final weighted pattern for each crystal is derived by summing the weighted values.
"""
"""
Understanding Self-Attention
Self-attention is a core concept in modern deep learning architectures, particularly transformers. It helps a model understand relationships between elements in a sequence by comparing each element with every other element.

Key Formula
The attention score between two elements 
i
i and 
j
j is calculated as:

Attention Score
i
,
j
=
Value
i
×
Value
j
Dimension
Attention Score 
i,j
​
 = 
Dimension
​
 
Value 
i
​
 ×Value 
j
​
 
​
 
Softmax Function
The softmax function converts raw attention scores into probabilities:

Softmax
(
x
i
)
=
e
x
i
∑
j
e
x
j
Softmax(x 
i
​
 )= 
∑ 
j
​
 e 
x 
j
​
 
 
e 
x 
i
​
 
 
​
 
Weighted Sum
Using the softmax scores, the final value for each element is calculated as a weighted sum:

Final Value
i
=
∑
j
Softmax Score
i
,
j
×
Value
j
Final Value 
i
​
 = 
j
∑
​
 Softmax Score 
i,j
​
 ×Value 
j
​
 
Example Calculation
Consider the following values:

Crystal values: 
[
4
,
2
,
7
,
1
,
9
]
[4,2,7,1,9]
Dimension: 
1
1
Step 1: Calculate Attention Scores
For crystal 
i
=
1
i=1 (
4
4):

Score
1
,
1
=
4
×
4
1
=
16
,
Score
1
,
2
=
4
×
2
1
=
8
,
…
Score 
1,1
​
 = 
1
​
 
4×4
​
 =16,Score 
1,2
​
 = 
1
​
 
4×2
​
 =8,…
Step 2: Apply Softmax
Convert scores to probabilities using softmax.

Step 3: Compute Weighted Sum
Multiply probabilities by crystal values and sum them to get the final value.

Applications
Self-attention is widely used in:

Natural Language Processing (e.g., transformers)
Computer Vision (e.g., Vision Transformers)
Sequence Analysis
Mastering self-attention provides a foundation for understanding advanced AI architectures.

"""
import numpy as np

def softmax(values):
    exps = np.exp(values - np.max(values))
    return exps / np.sum(exps)

def pattern_weaver(n, crystal_values, dimension):
    dimension_sqrt = np.sqrt(dimension)
    final_patterns = []

    for i in range(n):
        attention_scores = []
        for j in range(n):
            score = crystal_values[i] * crystal_values[j] / dimension_sqrt
            attention_scores.append(score)

        softmax_scores = softmax(attention_scores)
        weighted_sum = sum(softmax_scores[k] * crystal_values[k] for k in range(n))
        final_patterns.append(round(weighted_sum, 4))

    return final_patterns
