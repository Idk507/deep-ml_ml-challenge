"""
Implement an efficient method to update the mean reward for a k-armed bandit action after receiving each new reward, without storing the full history of rewards. Given the previous mean estimate (Q_prev), the number of times the action has been selected (k), and a new reward (R), compute the updated mean using the incremental formula.

Note: Using a regular mean that stores all past rewards will eventually run out of memory. Your solution should use only the previous mean, the count, and the new reward.

Example:
Input:
Q_prev = 2.0
k = 2
R = 6.0
new_Q = incremental_mean(Q_prev, k, R)
print(round(new_Q, 2))
Output:
4.0
Reasoning:
The updated mean is Q_prev + (1/k) * (R - Q_prev) = 2.0 + (1/2)*(6.0 - 2.0) = 2.0 + 2.0 = 4

Incremental Mean Update Rule
The incremental mean formula lets you update your estimate of the mean after each new observation, without keeping all previous rewards in memory. For the k-th reward 
R
k
R 
k
​
  and previous estimate 
Q
k
Q 
k
​
 :

Q
k
+
1
=
Q
k
+
1
k
(
R
k
−
Q
k
)
Q 
k+1
​
 =Q 
k
​
 + 
k
1
​
 (R 
k
​
 −Q 
k
​
 )
This saves memory compared to the regular mean, which requires storing all past rewards and recalculating each time. The incremental rule is crucial for online learning and large-scale problems where storing all data is impractical.
"""
def incremental_mean(Q_prev, k, R):
    """
    Q_prev: previous mean estimate (float)
    k: number of times the action has been selected (int)
    R: new observed reward (float)
    Returns: new mean estimate (float)
    """
    return Q_prev + (1 / k) * (R - Q_prev)
