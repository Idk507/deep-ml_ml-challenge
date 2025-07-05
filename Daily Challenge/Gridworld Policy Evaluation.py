"""
Implement policy evaluation for a 5x5 gridworld. Given a policy (mapping each state to action probabilities), compute the state-value function 
V
(
s
)
V(s) for each cell using the Bellman expectation equation. The agent can move up, down, left, or right, receiving a constant reward of -1 for each move. Terminal states (the four corners) are fixed at 0. Iterate until the largest change in 
V
V is less than a given threshold. Only use Python built-ins and no external RL libraries.

Example:
Input:
policy = {(i, j): {'up': 0.25, 'down': 0.25, 'left': 0.25, 'right': 0.25} for i in range(5) for j in range(5)}
gamma = 0.9
threshold = 0.001
V = gridworld_policy_evaluation(policy, gamma, threshold)
print(round(V[2][2], 4))
Output:
-7.0902
Reasoning:
The policy is uniform (equal chance of each move). The agent receives -1 per step. After iterative updates, the center state value converges to about -7.09, and corners remain at 0.
Gridworld Policy Evaluation
In reinforcement learning, policy evaluation is the process of computing the state-value function for a given policy. For a gridworld environment, this involves iteratively updating the value of each state based on the expected return following the policy.

Key Concepts
State-Value Function (V):
The expected return when starting from a state and following a given policy.

Policy:
A mapping from states to probabilities of selecting each available action.

Bellman Expectation Equation:
For each state 
s
s:

V
(
s
)
=
∑
a
π
(
a
∣
s
)
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
[
R
(
s
,
a
,
s
′
)
+
γ
V
(
s
′
)
]
V(s)= 
a
∑
​
 π(a∣s) 
s 
′
 
∑
​
 P(s 
′
 ∣s,a)[R(s,a,s 
′
 )+γV(s 
′
 )]
where:

π
(
a
∣
s
)
π(a∣s) is the probability of taking action 
a
a in state 
s
s,
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
 ∣s,a) is the probability of transitioning to state 
s
′
s 
′
 ,
R
(
s
,
a
,
s
′
)
R(s,a,s 
′
 ) is the reward for that transition,
γ
γ is the discount factor.
Algorithm Overview
Initialization:
Start with an initial guess (commonly zeros) for the state-value function 
V
(
s
)
V(s).

Iterative Update:
For each non-terminal state, update the state value using the Bellman expectation equation. Continue updating until the maximum change in value (delta) is less than a given threshold.

Terminal States:
For this example, the four corners of the grid are considered terminal, so their values remain unchanged.

This evaluation method is essential for understanding how "good" each state is under a specific policy, and it forms the basis for more advanced reinforcement learning algorithms.
"""
def gridworld_policy_evaluation(policy: dict, gamma: float, threshold: float) -> list[list[float]]:
    grid_size = 5
    V = [[0.0 for _ in range(grid_size)] for _ in range(grid_size)]
    actions = {'up': (-1, 0), 'down': (1, 0), 'left': (0, -1), 'right': (0, 1)}
    reward = -1
    while True:
        delta = 0.0
        new_V = [row[:] for row in V]
        for i in range(grid_size):
            for j in range(grid_size):
                if (i, j) in [(0, 0), (0, grid_size-1), (grid_size-1, 0), (grid_size-1, grid_size-1)]:
                    continue
                v = 0.0
                for action, prob in policy[(i, j)].items():
                    di, dj = actions[action]
                    ni = i + di if 0 <= i + di < grid_size else i
                    nj = j + dj if 0 <= j + dj < grid_size else j
                    v += prob * (reward + gamma * V[ni][nj])
                new_V[i][j] = v
                delta = max(delta, abs(V[i][j] - new_V[i][j]))
        V = new_V
        if delta < threshold:
            break
    return V
