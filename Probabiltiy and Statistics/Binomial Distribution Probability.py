"""
Binomial Distribution Probability

Write a Python function to calculate the probability of achieving exactly k successes in n independent Bernoulli trials, each with probability p of success, using the Binomial distribution formula.

Example:
Input:
n = 6, k = 2, p = 0.5
Output:
0.23438
Reasoning:
The function calculates the Binomial probability, the intermediate steps include calculating the binomial coefficient, raising p and (1-p) to the appropriate powers, and multiplying the results.

"""

import math

def binomial_probability(n, k, p):
    binomial_coeff = math.comb(n, k)
    probability = binomial_coeff * (p ** k) * ((1 - p) ** (n - k))
    return round(probability, 5)
