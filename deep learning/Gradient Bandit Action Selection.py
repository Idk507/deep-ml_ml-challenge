"""

Implement the gradient bandit algorithm for action selection in a multi-armed bandit setting. Write a class GradientBandit that maintains a set of action preferences and updates them after each reward. The class should provide a method select_action() to sample an action using the softmax of preferences, and a method update(action, reward) to update preferences using the gradient ascent update rule. Use only NumPy.

Example:
Input:
import numpy as np
gb = GradientBandit(num_actions=3, alpha=0.1)
a = gb.select_action()
gb.update(a, reward=1.0)
probs = gb.softmax()
print(np.round(probs, 2).tolist())
Output:
[0.32, 0.34, 0.34]
Reasoning:
After a positive reward, the selected action's preference is increased, boosting its softmax probability.
Gradient Bandits
Gradient Bandit algorithms are a family of action-selection methods for multi-armed bandit problems. Instead of estimating action values, they maintain a set of preferences for each action and use these to generate a probability distribution over actions via the softmax function. The algorithm then updates these preferences directly to increase the likelihood of selecting actions that yield higher rewards.

Algorithm Outline
Preferences (
H
a
H 
a
​
 ): For each action 
a
a, keep a real-valued preference 
H
a
H 
a
​
  (initialized to zero).
Action Probabilities (Softmax): At each timestep, choose action 
a
a with probability:
P
(
a
)
=
e
H
a
∑
j
e
H
j
P(a)= 
∑ 
j
​
 e 
H 
j
​
 
 
e 
H 
a
​
 
 
​
 
Preference Update Rule: After receiving reward 
R
t
R 
t
​
  for selected action 
A
t
A 
t
​
 , update preferences as:
H
a
←
H
a
+
α
⋅
(
R
t
−
R
t
ˉ
)
⋅
(
1
−
P
(
a
)
)
,
 if 
a
=
A
t
H 
a
​
 ←H 
a
​
 +α⋅(R 
t
​
 − 
R 
t
​
 
ˉ
​
 )⋅(1−P(a)), if a=A 
t
​
 
H
a
←
H
a
−
α
⋅
(
R
t
−
R
t
ˉ
)
⋅
P
(
a
)
,
 if 
a
≠
A
t
H 
a
​
 ←H 
a
​
 −α⋅(R 
t
​
 − 
R 
t
​
 
ˉ
​
 )⋅P(a), if a

=A 
t
​
 
Where:

R
t
ˉ
R 
t
​
 
ˉ
​
  is the running average reward (baseline, helps reduce variance)
α
α is the step size
Key Properties
Uses softmax probabilities for exploration (all actions get non-zero probability)
Action preferences directly drive probability updates
The baseline 
R
t
ˉ
R 
t
​
 
ˉ
​
  stabilizes learning and reduces update variance
More likely to select actions with higher expected reward
When to Use Gradient Bandits?
Problems where the best action changes over time (non-stationary)
Situations requiring continuous, adaptive exploration
Settings where value estimates are unreliable or less stable
Summary
Gradient bandit methods offer a principled way to learn action preferences by maximizing expected reward via gradient ascent. Their use of the softmax function ensures robust, probabilistic exploration and efficient learning from feedback.
"""
import numpy as np

class GradientBandit:
    def __init__(self, num_actions, alpha=0.1):
        """
        num_actions (int): Number of possible actions
        alpha (float): Step size for preference updates
        """
        self.num_actions = num_actions
        self.alpha = alpha
        self.preferences = np.zeros(num_actions)
        self.avg_reward = 0.0
        self.time = 0

    def softmax(self):
        exp_prefs = np.exp(self.preferences - np.max(self.preferences))  # for numerical stability
        return exp_prefs / np.sum(exp_prefs)

    def select_action(self):
        probs = self.softmax()
        return np.random.choice(self.num_actions, p=probs)

    def update(self, action, reward):
        self.time += 1
        self.avg_reward += (reward - self.avg_reward) / self.time
        probs = self.softmax()
        for a in range(self.num_actions):
            if a == action:
                self.preferences[a] += self.alpha * (reward - self.avg_reward) * (1 - probs[a])
            else:
                self.preferences[a] -= self.alpha * (reward - self.avg_reward) * probs[a]

