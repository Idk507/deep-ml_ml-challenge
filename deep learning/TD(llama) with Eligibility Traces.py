"""
Implement TD(llama) prediction with eligibility traces for estimating state values. TD(Î») unifies TD(0) and Monte Carlo methods through a decay parameter Î». Eligibility traces track which states are 'eligible' for learningârecently visited states get more credit for the current TD error. When Î»=0, this reduces to TD(0); when Î»=1, it approximates Monte Carlo. Your task is to implement the backward view of TD(Î») using accumulating eligibility traces.

Example:
Input:
episode=[(0,1), (1,1), (2,0)], gamma=0.9, lambda=0.8, alpha=0.1
Output:
States 0,1,2 updated with decaying eligibility
Reasoning:
At each step, we compute TD error δ, update eligibility traces (decay old traces, increment current state), then update all state values proportionally to their eligibility: V(s) += α * δ * e(s).

Learn About topic
Understanding TD(Î») with Eligibility Traces
TD(Î») is one of the most important algorithms in reinforcement learning. It provides a smooth interpolation between TD(0) and Monte Carlo methods, controlled by a single parameter Î» â [0,1].

The Problem with TD(0)
TD(0) updates only the immediately preceding state when a reward is observed. If a sequence of states leads to a reward, it takes many episodes for that credit to propagate back through all the states.

The Problem with Monte Carlo
Monte Carlo waits until episode end and updates all visited states. This is slowâwe can't learn until the episode finishesâand has high variance.

Eligibility Traces: The Best of Both Worlds
Eligibility traces maintain a memory of which states were recently visited. When we observe a TD error, we update all states in proportion to their eligibility.

The eligibility trace for state 
s
s at time 
t
t is:

e
t
(
s
)
=
γ
λ
 
e
t
−
1
(
s
)
+
1
(
S
t
=
s
)
e 
t
​
 (s)=γλe 
t−1
​
 (s)+1(S 
t
​
 =s)

where 
1
(
S
t
=
s
)
1(S 
t
​
 =s) is 1 if we're in state 
s
s at time 
t
t, else 0.

This says: decay the old eligibility by 
γ
λ
γλ, then add 1 if we just visited the state.

The TD Error
The TD error at time 
t
t is the same as in TD(0):

δ
t
=
R
t
+
1
+
γ
V
(
S
t
+
1
)
−
V
(
S
t
)
δ 
t
​
 =R 
t+1
​
 +γV(S 
t+1
​
 )−V(S 
t
​
 )

This measures how surprising the transition was.

The Update Rule
With eligibility traces, we update every state at every time step:

V
(
s
)
←
V
(
s
)
+
α
 
δ
t
 
e
t
(
s
)
∀
s
V(s)←V(s)+αδ 
t
​
 e 
t
​
 (s)∀s

States with high eligibility get large updates; states with zero eligibility are unchanged.

The Role of Î»
Î» = 0: Eligibility traces decay instantly. Only 
S
t
S 
t
​
  has non-zero eligibility, so we update only the current state. This is exactly TD(0).

Î» = 1: Eligibility traces decay only by 
γ
γ. All visited states maintain significant eligibility. With 
γ
=
1
γ=1, this is equivalent to Monte Carlo.

0 < Î» < 1: A smooth tradeoff. Recent states get more credit than distant states, but credit still propagates back multiple steps.

Forward vs Backward View
There are two equivalent ways to understand TD(Î»):

Forward View: Each state's update uses the Î»-return, a weighted average of n-step returns:

G
t
λ
=
(
1
−
λ
)
∑
n
=
1
∞
λ
n
−
1
G
t
(
n
)
G 
t
λ
​
 =(1−λ)∑ 
n=1
∞
​
 λ 
n−1
 G 
t
(n)
​
 

Backward View: At each step, compute TD error and update all states by their eligibility. This is what we implementâit's more practical for online learning.

The two views produce identical updates (for offline/batch updates).

Algorithm Summary
Initialize 
V
(
s
)
=
0
V(s)=0 for all states, 
e
(
s
)
=
0
e(s)=0 for all states.

For each step in the episode:

Observe reward 
R
R and next state 
S
′
S 
′
 
Compute TD error: 
δ
=
R
+
γ
V
(
S
′
)
−
V
(
S
)
δ=R+γV(S 
′
 )−V(S)
Update eligibility: 
e
(
S
)
←
e
(
S
)
+
1
e(S)←e(S)+1 (after decaying all traces)
For all states: 
V
(
s
)
←
V
(
s
)
+
α
δ
 
e
(
s
)
V(s)←V(s)+αδe(s)
Decay all traces: 
e
(
s
)
←
γ
λ
 
e
(
s
)
e(s)←γλe(s)
S
←
S
′
S←S 
′
 
Accumulating vs Replacing Traces
Accumulating traces (what we implement): 
e
(
s
)
←
γ
λ
 
e
(
s
)
+
1
e(s)←γλe(s)+1

If we visit the same state twice, eligibility accumulates.

Replacing traces: 
e
(
s
)
←
1
e(s)←1 on visit, regardless of old value.

Replacing traces often work better in practice but accumulating traces have cleaner theory.

Practical Benefits
TD(Î») with Î» â 0.8-0.9 often learns faster than both TD(0) and Monte Carlo. The eligibility traces allow credit to propagate back multiple steps immediately, rather than waiting for many episodes.


"""

import numpy as np

def td_lambda_prediction(
    episodes: list[list[tuple[int, float]]],
    n_states: int,
    gamma: float,
    lambd: float,
    alpha: float
) -> np.ndarray:
    """
    Estimate state values using TD(λ) with accumulating eligibility traces.
    
    Args:
        episodes: List of episodes. Each episode is a list of (state, reward) tuples.
                 The reward at index i is the reward received AFTER leaving state i.
        n_states: Number of states (states are integers 0 to n_states-1)
        gamma: Discount factor
        lambd: Trace decay parameter (λ). Use 'lambd' to avoid Python keyword.
        alpha: Learning rate
        
    Returns:
        V: Estimated state values as numpy array of shape (n_states,)
    """
    # Initialize value function and eligibility traces
    V = np.zeros(n_states)
    
    # Process each episode
    for episode in episodes:
        # Reset eligibility traces at the start of each episode
        e = np.zeros(n_states)
        
        # Process each step in the episode
        for t in range(len(episode)):
            state, reward = episode[t]
            
            # Determine next state and its value
            if t + 1 < len(episode):
                next_state = episode[t + 1][0]
                next_value = V[next_state]
            else:
                # Terminal state: next value is 0
                next_value = 0.0
            
            # Compute TD error
            delta = reward + gamma * next_value - V[state]
            
            # Update eligibility trace for current state (accumulating)
            e[state] += 1
            
            # Update all state values proportional to their eligibility
            V += alpha * delta * e
            
            # Decay all eligibility traces for next step
            e *= gamma * lambd
    
    return V


