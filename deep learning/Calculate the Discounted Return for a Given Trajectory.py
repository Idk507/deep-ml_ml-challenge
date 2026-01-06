"""
Calculate the Discounted Return for a Given Trajectory
Easy
Reinforcement Learning

Write a function that computes the discounted return 
G
t
=
∑
k
=
0
∞
γ
k
R
t
+
k
+
1
G 
t
​
 =∑ 
k=0
∞
​
 γ 
k
 R 
t+k+1
​
  for a given sequence of rewards and discount factor gamma. This quantity corresponds to the expected return 
v
π
(
s
)
v 
π
​
 (s) in reinforcement learning, as defined by the equation in the image. Only use NumPy.

Example:
Input:
rewards = [1, 2, 3, 4]
gamma = 0.9
print(discounted_return(rewards, gamma))
Output:
8.146
Reasoning:
G = 1 + 0.92 + 0.9^23 + 0.9^3*4 = 1 + 1.8 + 2.43 + 2.916 = 8.146

Learn About topic
Discounted Return in Reinforcement Learning
The discounted return 
G
t
G 
t
​
  is the total expected sum of rewards that an agent will receive in the future, where future rewards are multiplied by a discount factor 
γ
γ (
0
≤
γ
≤
1
0≤γ≤1). This concept is central in reinforcement learning and is captured by:

G
t
=
∑
k
=
0
∞
γ
k
R
t
+
k
+
1
G 
t
​
 = 
k=0
∑
∞
​
 γ 
k
 R 
t+k+1
​
 
G
t
G 
t
​
 : Discounted return at time 
t
t
R
t
+
k
+
1
R 
t+k+1
​
 : Reward received 
k
k steps into the future
γ
γ: Discount factor (controls how much the agent cares about future rewards)
Why Discount?
Ensures the sum converges (for infinite horizons)
Encourages the agent to prefer immediate rewards over distant rewards (if 
γ
<
1
γ<1)
Models uncertainty or time preference
Connection to Value Function
The state-value function 
v
π
(
s
)
v 
π
​
 (s) under policy 
π
π is the expected discounted return when starting in state 
s
s and following 
π
π:

v
π
(
s
)
=
E
π
[
G
t
∣
S
t
=
s
]
v 
π
​
 (s)=E 
π
​
 [G 
t
​
 ∣S 
t
​
 =s]
This means the value of a state is the expected sum of discounted future rewards starting from that state.

Summary
Discounted return quantifies the future reward an agent expects, accounting for time and uncertainty
It is foundational for defining optimality and learning in reinforcement learning
Implementing the sum is often the first step to understanding policy evaluation


"""


import numpy as np

def discounted_return(rewards, gamma):
    """
    Compute the discounted return for a given list of rewards.
    
    Args:
        rewards (list or np.ndarray): sequence of rewards R_{t+1}, R_{t+2}, ...
        gamma (float): discount factor (0 <= gamma <= 1)
    
    Returns:
        float: discounted return G_t
    """
    rewards = np.array(rewards, dtype=float)
    # Create discount factors [gamma^0, gamma^1, ..., gamma^(n-1)]
    discounts = np.power(gamma, np.arange(len(rewards)))
    # Compute sum of discounted rewards
    return np.sum(discounts * rewards)  
