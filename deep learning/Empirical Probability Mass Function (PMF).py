"""
Problem
Given a list of integer samples drawn from a discrete distribution, implement a function to compute the empirical Probability Mass Function (PMF). The function should return a list of (value, probability) pairs sorted by the value in ascending order. If the input is empty, return an empty list.

Example:
Input:
samples = [1, 2, 2, 3, 3, 3]
Output:
[(1, 0.16666666666666666), (2, 0.3333333333333333), (3, 0.5)]
Reasoning:
Counts are {1:1, 2:2, 3:3} over 6 samples, so probabilities are 1/6, 2/6, and 3/6 respectively, returned sorted by value.

Learn Section
Probability Mass Function (PMF) â Simple Explanation
A probability mass function (PMF) describes how probabilities are assigned to the possible outcomes of a discrete random variable.

It tells you the chance of each specific outcome.
Each probability is non-negative.
The total of all probabilities adds up to 1.
Estimating from data
If the true probabilities are unknown, you can estimate them with an empirical PMF:

Count how often each outcome appears.
Divide by the total number of observations.
Example
Observed sequence: 1, 2, 2, 3, 3, 3 (6 outcomes total)

â1â appears once â estimated probability = 1/6
â2â appears twice â estimated probability = 2/6 = 1/3
â3â appears three times â estimated probability = 3/6 = 1/2
"""

from collections import Counter

def empirical_pmf(samples):
    """
    Given an iterable of integer samples, return a list of (value, probability)
    pairs sorted by value ascending.
    """
    if not samples:  # Case: empty input
        return []
    
    total = len(samples)
    counts = Counter(samples)  # Count occurrences of each value
    
    # Compute probabilities and sort by value
    pmf = [(value, counts[value] / total) for value in sorted(counts)]
    
    return pmf


