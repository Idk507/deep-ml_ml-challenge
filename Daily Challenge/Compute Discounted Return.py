"""
Compute Discounted Return
Easy
Reinforcement Learning

Write a function to compute the discounted return for a sequence of rewards given a discount factor gamma. The function should take a list or NumPy array of rewards and a discount factor gamma (0 < gamma <= 1) and return the scalar value of the total discounted return. Only use NumPy.

Example:
Input:
rewards = [1, 1, 1]
gamma = 0.5
print(discounted_return(rewards, gamma))
Output:
1.75
Reasoning:
Discounted return: 11 + 10.5 + 1*0.25 = 1 + 0.5 + 0.25 = 1.75

Learn About topic
Discounted Return in Reinforcement Learning
The discounted return is a core concept in reinforcement learning that quantifies the total expected reward an agent can achieve from a sequence of actions, while accounting for the fact that immediate rewards are typically more valuable than distant future rewards.

Definition
Given a sequence of rewards 
[
r
0
,
r
1
,
…
,
r
T
−
1
]
[r 
0
​
 ,r 
1
​
 ,…,r 
T−1
​
 ] and a discount factor 
γ
γ (
0
<
γ
≤
1
0<γ≤1), the discounted return 
G
G is:

G
=
r
0
+
γ
r
1
+
γ
2
r
2
+
⋯
+
γ
T
−
1
r
T
−
1
G=r 
0
​
 +γr 
1
​
 +γ 
2
 r 
2
​
 +⋯+γ 
T−1
 r 
T−1
​
 
Or, more generally:

G
=
∑
t
=
0
T
−
1
γ
t
r
t
G= 
t=0
∑
T−1
​
 γ 
t
 r 
t
​
 
r
t
r 
t
​
 : reward at time step 
t
t
γ
γ: discount factor (how much future rewards are âworthâ relative to immediate rewards)
Why Discount Future Rewards?
Uncertainty: Future rewards are less certain than immediate rewards.
Preference: Most decision-makers prefer immediate rewards over delayed ones.
Mathematical Convenience: Ensures the sum converges even for infinite sequences (if 
γ
<
1
γ<1).
Properties
If 
γ
=
1
γ=1, the agent values all rewards equally (undiscounted return).
If 
γ
=
0.9
γ=0.9, the agent heavily favors immediate rewards over distant ones.
The smaller 
γ
γ is, the less impact future rewards have on the total return.
Practical Example
Suppose 
r
e
w
a
r
d
s
=
[
2
,
0
,
0
,
3
]
rewards=[2,0,0,3] and 
γ
=
0.5
γ=0.5:

G
=
2
∗
1
+
0
∗
0.5
+
0
∗
0.25
+
3
∗
0.125
=
2
+
0
+
0
+
0.375
=
2.375
G=2∗1+0∗0.5+0∗0.25+3∗0.125=2+0+0+0.375=2.375
Summary
The discounted return captures the âpresent valueâ of a sequence of rewards, providing a crucial foundation for learning and planning in reinforcement learning.

"""
import numpy as np

def discounted_return(rewards, gamma):
    rewards = np.array(rewards, dtype=np.float32)
    time_steps = np.arange(len(rewards))
    discounts = gamma ** time_steps
    return np.sum(rewards * discounts)
