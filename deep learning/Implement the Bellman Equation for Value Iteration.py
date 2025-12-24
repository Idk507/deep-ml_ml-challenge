import numpy as np

def bellman_update(V, transitions, gamma):
    """
    Perform one step of value iteration using the Bellman equation.
    
    Args:
      V: np.ndarray, state values, shape (n_states,)
      transitions: list of dicts. transitions[s][a] is a list of (prob, next_state, reward, done)
      gamma: float, discount factor
      
    Returns:
      np.ndarray, updated state values
    """
    n_states = len(transitions)
    new_V = np.zeros_like(V)

    for s in range(n_states):
        action_values = []
        for a, outcomes in transitions[s].items():
            q_val = 0.0
            for prob, next_state, reward, done in outcomes:
                if done:
                    q_val += prob * reward
                else:
                    q_val += prob * (reward + gamma * V[next_state])
            action_values.append(q_val)
        new_V[s] = np.max(action_values) if action_values else 0.0

    return new_V
