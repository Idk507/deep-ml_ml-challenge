"""
Implement the epsilon-greedy method for action selection in an n-armed bandit problem. Given a set of estimated action values (Q-values), select an action using the epsilon-greedy policy: with probability epsilon, choose a random action; with probability 1 - epsilon, choose the action with the highest estimated value.

Example:
Input:
Q = np.array([0.5, 2.3, 1.7])
epsilon = 0.0
action = epsilon_greedy(Q, epsilon)
print(action)
Output:
1
Reasoning:
With epsilon=0.0 (always greedy), the highest Q-value is 2.3 at index 1, so the function always returns 1.
Epsilon-Greedy Policy
The epsilon-greedy method is a fundamental action selection strategy used in reinforcement learning, especially for solving the n-armed bandit problem. The key idea is to balance exploration (trying new actions) and exploitation (choosing the best-known action):

With probability 
ε
ε (epsilon), the agent explores by selecting an action at random.
With probability 
1
−
ε
1−ε, it exploits by choosing the action with the highest estimated value (greedy choice).
The epsilon-greedy policy is simple to implement and provides a way to avoid getting stuck with suboptimal actions due to insufficient exploration.
"""
import numpy as np

def epsilon_greedy(Q, epsilon=0.1):
    """
    Selects an action using epsilon-greedy policy.
    Q: np.ndarray of shape (n,) -- estimated action values
    epsilon: float in [0, 1]
    Returns: int, selected action index
    """
    if np.random.rand() < epsilon:
        # Explore: choose a random action
        return np.random.randint(len(Q))
    else:
        # Exploit: choose the action with the highest estimated value
        return np.argmax(Q)
