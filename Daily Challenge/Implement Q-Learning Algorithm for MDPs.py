"""
Write a function that implements the Q-Learning algorithm to learn the optimal Q-table for a given Markov Decision Process (MDP). The function should take the number of states, number of actions, transition probabilities matrix, rewards matrix, list of terminal states, learning rate, discount factor, epsilon for exploration, and the number of episodes as inputs. Use these parameters to iteratively update the Q-table based on the Q-Learning update rule, employing an epsilon-greedy strategy for action selection. Ensure the function handles starting from non-terminal states and stops episodes upon reaching a terminal state.

Constraints:

num_states: Integer greater than or equal to 1.
num_actions: Integer greater than or equal to 1.
P: A 3D NumPy array of shape (num_states, num_actions, num_states) where each element is a probability between 0 and 1, and each sub-array sums to 1.
R: A 2D NumPy array of shape (num_states, num_actions) with float or integer values.
terminal_states: A list or NumPy array of integers, each between 0 and num_states - 1, with no duplicates.
alpha: A float between 0 and 1.
gamma: A float between 0 and 1.
epsilon: A float between 0 and 1.
num_episodes: An integer greater than or equal to 1. The function should return a 2D NumPy array of shape (num_states, num_actions) representing the learned Q-table.
Example:
Input:
import numpy as np; np.random.seed(42); P = np.array([[[0, 1], [1, 0]], [[1, 0], [1, 0]]]); R = np.array([[1, 0], [0, 0]]); terminal_states = [1]; print(q_learning(2, 2, P, R, terminal_states, 0.1, 0.9, 0.1, 10))
Output:
[[0.65132156, 0.052902  ],[0., 0.]]
Reasoning:
The Q-Learning algorithm initializes a Q-table with zeros and iteratively updates it over 10 episodes by starting from random non-terminal states, selecting actions via an epsilon-greedy policy, sampling next states and rewards from the provided transition probabilities (P) and rewards (R), and applying the update rule: Q(s, a) += alpha * (reward + gamma * max(Q(next_state)) - Q(s, a)). This process results in the output Q-table [[0.65132156, 0.052902], [0., 0.]], where the values represent learned estimates of state-action values, with the second state's Q-values remaining zero because it is a terminal state and no further actions are taken from there.
Q-Learning: Learning Optimal Actions in Markov Decision Processes
Q-Learning is a method in reinforcement learning used to estimate the value of taking specific actions in different states within a Markov Decision Process (MDP). An MDP models decision-making scenarios where the outcomes of actions depend on the current state, and the goal is to maximize long-term rewards. This section breaks down the key concepts step by step, focusing on the underlying mathematics.

1. Understanding Markov Decision Processes
A Markov Decision Process is a framework for sequential decision-making. It consists of states, actions, transition probabilities, and rewards. In an MDP, the future state depends only on the current state and the chosen action, not on the history of previous states.

States represent the situations an agent might encounter.
Actions are the choices available in each state.
Transition probabilities describe the likelihood of moving from one state to another after an action.
Rewards are numerical values that quantify the immediate benefit of taking an action in a state.
For example, imagine navigating a simple grid where each cell is a state, moving right or left is an action, and reaching a goal gives a reward.

2. The Q-Value Function
At the heart of Q-Learning is the Q-value, which estimates the total expected reward of taking a specific action in a given state and then following the best possible strategy afterward.

Mathematically, the Q-value for a state 
s
s and action 
a
a is denoted as 
Q
(
s
,
a
)
Q(s,a). It is defined by the equation:

Q
(
s
,
a
)
=
r
(
s
,
a
)
+
γ
∑
s
′
P
(
s
′
∣
s
,
a
)
max
⁡
a
′
Q
(
s
′
,
a
′
)
Q(s,a)=r(s,a)+γ 
s 
′
 
∑
​
 P(s 
′
 ∣s,a) 
a 
′
 
max
​
 Q(s 
′
 ,a 
′
 )
Here:

r
(
s
,
a
)
r(s,a) is the immediate reward received for taking action 
a
a in state 
s
s.
γ
γ (gamma) is the discount factor, a number between 0 and 1 that reduces the importance of future rewards over time (e.g., if 
γ
=
0.9
γ=0.9, rewards in the near future are valued more than those far ahead).
P
(
s
′
∣
s
,
a
)
P(s 
′
 ∣s,a) is the transition probability, representing the likelihood of ending up in state 
s
′
s 
′
  after action 
a
a in state 
s
s.
max
⁡
a
′
Q
(
s
′
,
a
′
)
max 
a 
′
 
​
 Q(s 
′
 ,a 
′
 ) is the maximum Q-value of all possible actions in the next state 
s
′
s 
′
 , indicating the best future choice.
This equation captures the idea that the Q-value balances immediate rewards with the discounted value of future rewards, helping to identify the most valuable actions over time.

3. The Q-Learning Update Rule
Q-Learning updates the Q-value estimates iteratively based on experience, using a simple iterative formula. This process allows the agent to learn from trials without needing to know the full transition probabilities in advance.

The update rule is:

Q
(
s
,
a
)
←
Q
(
s
,
a
)
+
α
[
r
+
γ
max
⁡
a
′
Q
(
s
′
,
a
′
)
−
Q
(
s
,
a
)
]
Q(s,a)←Q(s,a)+α[r+γ 
a 
′
 
max
​
 Q(s 
′
 ,a 
′
 )−Q(s,a)]
In this equation:

α
α (alpha) is the learning rate, a value between 0 and 1 that controls how much new information overrides old estimates (e.g., if 
α
=
0.1
α=0.1, updates are gradual).
r
r is the reward observed after taking action 
a
a in state 
s
s.
s
′
s 
′
  is the next state that results from the action.
The term inside the brackets, 
r
+
γ
max
⁡
a
′
Q
(
s
′
,
a
′
)
−
Q
(
s
,
a
)
r+γmax 
a 
′
 
​
 Q(s 
′
 ,a 
′
 )−Q(s,a), is the difference between the estimated Q-value and the actual experienced value, known as the temporal difference error.
This rule refines Q-values over multiple episodes, gradually converging to the optimal values that maximize long-term rewards.

4. Balancing Exploration and Exploitation
To learn effectively, Q-Learning must balance exploring new actions (to discover potential rewards) and exploiting known high-value actions.

This is achieved through an epsilon-greedy strategy, where:

With probability 
ϵ
ϵ (epsilon, a small number like 0.1), a random action is selected to encourage exploration.
With probability 
1
−
ϵ
1−ϵ, the action with the highest Q-value is chosen to exploit current knowledge.
For instance, if 
ϵ
=
0.2
ϵ=0.2, in 20% of decisions, the agent tries something random, while in 80%, it picks the best-known option. Over time, 
ϵ
ϵ can be reduced to favor exploitation as learning progresses.

Example Walkthrough
Consider a simple two-state MDP: State A and State B, with two actions in each (Action 1 and Action 2). Suppose:

From State A, Action 1 leads to State B with probability 1 and a reward of 1.
From State B, any action ends the process with a reward of 0 (State B is terminal).
Let 
γ
=
0.9
γ=0.9 and 
α
=
0.5
α=0.5.
Initially, assume all Q-values are 0. In the first episode:

Start in State A and choose Action 1 (greedily, since all Q-values are equal).
Move to State B, receive reward 1, and since State B is terminal, the update is:
Q
(
A
,
Action 1
)
←
0
+
0.5
[
1
+
0.9
⋅
0
−
0
]
=
0.5
Q(A,Action 1)←0+0.5[1+0.9⋅0−0]=0.5
Now, Q(A, Action 1) is 0.5, so in future episodes, Action 1 is more likely in State A.
Through repeated episodes, Q-values adjust to reflect the best long-term rewards, such as prioritizing paths that lead to higher cumulative rewards.
"""

import numpy as np
def q_learning(num_states, num_actions, P, R, terminal_states, alpha, gamma, epsilon, num_episodes):
    # Your code here
    q = np.zeros((num_states,num_actions))
    for episode in range(num_episodes):
        state = np.random.choice([s for s in range(num_states) if s not in terminal_states])
        while state not in terminal_states :
            if np.random.rand() < epsilon :
                action = np.random.randint(num_actions)
            else :
                action = np.argmax(q[state])
            next_state =  np.random.choice(num_states,p = P[state, action])

            reward = R[state,action]
            if next_state in terminal_states : 
                target = reward 
            else :
                target = reward + gamma * np.max(q[next_state])
            q[state, action] += alpha * (target - q[state, action])    
            state = next_state 
    return q

