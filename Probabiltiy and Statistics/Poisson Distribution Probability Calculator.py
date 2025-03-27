"""
Poisson Distribution Probability Calculator

Write a Python function to calculate the probability of observing exactly k events in a fixed interval using the Poisson distribution formula. The function should take k (number of events) and lam (mean rate of occurrences) as inputs and return the probability rounded to 5 decimal places.

Example:
Input:
k = 3, lam = 5
Output:
0.14037
Reasoning:
The function calculates the probability for a given number of events occurring in a fixed interval, based on the mean rate of occurrences.
"""
import math

def poisson_probability(k, lam):
	"""
	Calculate the probability of observing exactly k events in a fixed interval,
	given the mean rate of events lam, using the Poisson distribution formula.
	:param k: Number of events (non-negative integer)
	:param lam: The average rate (mean) of occurrences in a fixed interval
	"""
	# Your code here
	val = (lam**k)*math.exp(-lam)/math.factorial(k)
	return round(val,5)
