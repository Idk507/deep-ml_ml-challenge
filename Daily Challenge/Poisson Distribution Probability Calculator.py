"""
Write a Python function to calculate the probability of observing exactly k events in a fixed interval using the Poisson distribution formula. The function should take k (number of events) and lam (mean rate of occurrences) as inputs and return the probability rounded to 5 decimal places.

Example:
Input:
k = 3, lam = 5
Output:
0.14037
Reasoning:
The function calculates the probability for a given number of events occurring in a fixed interval, based on the mean rate of occurrences.

Understanding Poisson Distribution
The Poisson distribution is a discrete probability distribution that expresses the probability of a given number of events occurring in a fixed interval of time or space, provided these events occur with a known constant mean rate and independently of the time since the last event.

Mathematical Definition
The probability of observing ( k ) events in a given interval is defined as:

P
(
k
;
λ
)
=
λ
k
e
−
λ
k
!
P(k;λ)= 
k!
λ 
k
 e 
−λ
 
​
 
( k ): Number of events (non-negative integer)
( \lambda ): The mean number of events in the given interval (rate parameter)
( e ): Euler's number, approximately 2.718
Key Properties
Mean: ( \lambda )
Variance: ( \lambda )
The Poisson distribution is used for modeling rare or random events.
Example Calculation
Suppose the mean number of calls received in an hour (( \lambda )) is 5. Calculate the probability of receiving exactly 3 calls in an hour:

Substitute into the formula:

P
(
3
;
5
)
=
5
3
e
−
5
3
!
P(3;5)= 
3!
5 
3
 e 
−5
 
​
 
Calculate step-by-step:

P
(
3
;
5
)
=
125
⋅
e
−
5
6
≈
0.14037
P(3;5)= 
6
125⋅e 
−5
 
​
 ≈0.14037
Applications
The Poisson distribution is widely used in:

Modeling the number of arrivals at a queue (e.g., calls at a call center)
Counting occurrences over time (e.g., number of emails received per hour)
Biology (e.g., distribution of mutations in a DNA strand)
Traffic flow analysis (e.g., number of cars passing through an intersection)
This distribution is essential for understanding and predicting rare events in real-world scenarios.
https://www.deep-ml.com/problems/81
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
