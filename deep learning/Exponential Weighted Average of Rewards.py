"""
Exponential Weighted Average of Rewards
Given an initial value 
Q
1
Q 
1
​
 , a list of 
k
k observed rewards 
R
1
,
R
2
,
…
,
R
k
R 
1
​
 ,R 
2
​
 ,…,R 
k
​
 , and a step size 
α
α, implement a function to compute the exponentially weighted average as:

(
1
−
α
)
k
Q
1
+
∑
i
=
1
k
α
(
1
−
α
)
k
−
i
R
i
(1−α) 
k
 Q 
1
​
 +∑ 
i=1
k
​
 α(1−α) 
k−i
 R 
i
​
 

This weighting gives more importance to recent rewards, while the influence of the initial estimate 
Q
1
Q 
1
​
  decays over time. Do not use running/incremental updates; instead, compute directly from the formula. (This is called the exponential recency-weighted average.)

Example:
Input:
Q1 = 2.0
rewards = [5.0, 9.0]
alpha = 0.3
result = exp_weighted_average(Q1, rewards, alpha)
print(round(result, 4))
Output:
5.003
Reasoning:
Here, k=2, so the result is: (1-0.3)^22.0 + 0.3(1-0.3)^15.0 + 0.3(1-0.3)^09.0 = 0.492.0 + 0.215.0 + 0.39.0 = 0.98 + 1.05 + 2.7 = 4.73 (actually, should be 0.492+0.30.75+0.39 = 0.98+1.05+2.7=4.73)Exponential Recency-Weighted Average
When the environment is nonstationary, it is better to give more weight to recent rewards. The formula 
(
1
−
α
)
k
Q
1
+
∑
i
=
1
k
α
(
1
−
α
)
k
−
i
R
i
(1−α) 
k
 Q 
1
​
 +∑ 
i=1
k
​
 α(1−α) 
k−i
 R 
i
​
  computes the expected value by exponentially decaying the influence of old rewards and the initial estimate. The parameter 
α
α controls how quickly old information is forgotten: higher 
α
α gives more weight to new rewards.


"""

def exp_weighted_average(Q1, rewards, alpha):
    """
    Q1: float, initial estimate
    rewards: list or array of rewards, R_1 to R_k
    alpha: float, step size (0 < alpha <= 1)
    Returns: float, exponentially weighted average after k rewards
    """
    k = len(rewards)
    base = (1 - alpha) ** k
    weighted_sum = base * Q1
    for i in range(1, k + 1):
        weight = alpha * (1 - alpha) ** (k - i)
        weighted_sum += weight * rewards[i - 1]
    return weighted_sum
