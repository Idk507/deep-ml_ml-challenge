"""
A gambler has the chance to bet on a sequence of coin flips. If the coin lands heads, the gambler wins the amount staked; if tails, the gambler loses the stake. The goal is to reach 100, starting from a given capital 
s
s (with 
0
<
s
<
100
0<s<100). The game ends when the gambler reaches 
0
0 (bankruptcy) or 
100
100 (goal). On each flip, the gambler can bet any integer amount from 
1
1 up to 
min
⁡
(
s
,
100
−
s
)
min(s,100−s).

The probability of heads is 
p
h
p 
h
​
  (known). Reward is 
+
1
+1 if the gambler reaches 
100
100 in a transition, 
0
0 otherwise.

Your Task: Write a function gambler_value_iteration(ph, theta=1e-9) that:

Computes the optimal state-value function 
V
(
s
)
V(s) for all 
s
=
1
,
.
.
.
,
99
s=1,...,99 using value iteration.
Returns the optimal policy as a mapping from state 
s
s to the optimal stake 
a
∗
a 
∗
  (can return any optimal stake if there are ties).
Inputs:

ph: probability of heads (float between 0 and 1)
theta: threshold for value iteration convergence (default 
1
e
−
9
1e−9)
Returns:

V: array/list of length 101, 
V
[
s
]
V[s] is the value for state 
s
s
policy: array/list of length 101, 
p
o
l
i
c
y
[
s
]
policy[s] is the optimal stake in state 
s
s (0 if 
s
=
0
s=0 or 
s
=
100
s=100)
Example:
Input:
ph = 0.4
V, policy = gambler_value_iteration(ph)
print(round(V[50], 4))
print(policy[50])
Output:
0.0178
1
Reasoning:
From state 50, the optimal action is to bet 1, with a probability of reaching 100 of about 0.0178 when ph=0.4.


"""

def gambler_value_iteration(ph, theta=1e-9):
    # Initialize value function for states 0 to 100; terminal states 0 and 100 have value 0
    V = [0.0] * 101
    # Initialize policy array (bet amount for each state)
    policy = [0] * 101
    
    # Value iteration loop
    while True:
        delta = 0
        # Iterate over non-terminal states (1 to 99)
        for s in range(1, 100):
            # Possible actions: bet between 1 and min(s, 100 - s)
            actions = range(1, min(s, 100 - s) + 1)
            action_returns = []
            # Evaluate each action
            for a in actions:
                win_state = s + a
                lose_state = s - a
                # Reward is 1 if transition reaches 100, else 0
                reward = 1.0 if win_state == 100 else 0.0
                # Expected value: ph * (reward + V[win]) + (1 - ph) * V[lose]
                ret = ph * (reward + V[win_state]) + (1 - ph) * V[lose_state]
                action_returns.append(ret)
            # Update V[s] with the maximum expected value
            max_value = max(action_returns)
            delta = max(delta, abs(V[s] - max_value))
            V[s] = max_value
        # Check for convergence
        if delta < theta:
            break
    
    # Extract optimal policy
    for s in range(1, 100):
        actions = range(1, min(s, 100 - s) + 1)
        best_action = 0
        best_return = -float('inf')
        # Find action that maximizes expected value
        for a in actions:
            win_state = s + a
            lose_state = s - a
            reward = 1.0 if win_state == 100 else 0.0
            ret = ph * (reward + V[win_state]) + (1 - ph) * V[lose_state]
            if ret > best_return:
                best_return = ret
                best_action = a
        policy[s] = best_action
    
    return V, policy
