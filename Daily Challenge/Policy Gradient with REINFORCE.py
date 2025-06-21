"""
Policy Gradient with REINFORCE

Implement the policy gradient estimator using the REINFORCE algorithm. The policy is parameterized by a 2D NumPy array theta of shape (num_states, num_actions). The policy for each state is computed via softmax over theta[s, :]. Given a list of episodes (each a list of (state, action, reward) tuples), compute the average gradient of the log-policy multiplied by the return at each time step.

Example:
Input:
theta = np.zeros((2,2)); episodes = [[(0,1,0), (1,0,1)], [(0,0,0)]]
Output:
[[-0.25, 0.25], [0.25, -0.25]]
Reasoning:
Episode 1 contributes a positive gradient from reward 1 at t=1; episode 2 adds zero. Result is averaged across episodes.

"""

import numpy as np

def compute_policy_gradient(theta, episodes):
    def softmax(x):
        x = x - np.max(x)
        exps = np.exp(x)
        return exps / np.sum(exps)

    grad = np.zeros_like(theta)
    for episode in episodes:
        rewards = [step[2] for step in episode]
        returns = np.cumsum(rewards[::-1])[::-1]
        for t, (s, a, _), G in zip(range(len(episode)), episode, returns):
            probs = softmax(theta[s])
            grad_log_pi = np.zeros_like(theta)
            grad_log_pi[s, :] = -probs
            grad_log_pi[s, a] += 1
            grad += grad_log_pi * G
    return grad / len(episodes)
