"""
Implement the SARSA algorithm to estimate Q-values for a given set of deterministic transitions using greedy action selection.

All Q-values are initialized to zero.
Each episode starts from a given initial state.
The episode ends when it reaches the 
t
e
r
m
i
n
a
l
terminal state or when the number of steps exceeds 
m
a
x
s
t
e
p
s
maxsteps.
Changes made to Q-values are persistent across episodes.
Example:
Input:
transitions = {
    ('A', 'left'): (5.0, 'B'),
    ('A', 'right'): (1.0, 'C'),
    ('B', 'left'): (2.0, 'A'),
    ('B', 'right'): (0.0, 'C'),
    ('C', 'down'): (1.0, 'terminal')
}

initial_states = ['A', 'B']
alpha = 0.1
gamma = 0.9
max_steps = 10

Q = sarsa_update(transitions, initial_states, alpha, gamma, max_steps)

for k in sorted(transitions):
    print(f"Q{str(k):15} = {Q[k]:.4f}")
Output:
Q('A', 'left')   = 4.2181
Q('A', 'right')  = 0.0000
Q('B', 'left')   = 2.7901
Q('B', 'right')  = 0.0000
Reasoning:
The SARSA update rule is: Q(s,a) <- Q(s,a) + alpha * [reward + gamma * Q(s',a') - Q(s,a)]

Starting from initial Q-values of 0, each episode updates Q-values based on the transitions.

Q('A', 'left') increases because it leads to B, and B can eventually return to A or C with additional rewards.
Q('A', 'right') and Q('B', 'right') remain 0.0 because the next state C leads directly to terminal with small reward.
Q('B', 'left') increases due to cyclic transitions giving non-zero rewards.
Learn About topic
SARSA: On-Policy TD Control
Goal: Estimate the action-value function 
Q
π
≈
q
∗
Q 
π
 ≈q 
∗
  using the SARSA algorithm (on-policy Temporal-Difference control).

Parameters
Step size 
α
∈
(
0
,
1
]
α∈(0,1]
Discount factor 
γ
∈
[
0
,
1
]
γ∈[0,1]
Initialization
Initialize 
Q
(
s
,
a
)
Q(s,a) arbitrarily for all 
s
∈
S
+
s∈S 
+
 , 
a
∈
A
(
s
)
a∈A(s)
Set 
Q
(
terminal
,
⋅
)
=
0
Q(terminal,⋅)=0
Algorithm
Loop for each episode:

Initialize state 
S
S

Choose action 
A
A from 
S
S using a policy derived from 
Q
Q (e.g., greedy)

Loop for each step of the episode:

Take action 
A
A, observe reward 
R
R and next state 
S
′
S 
′
 
Choose next action 
A
′
A 
′
  from 
S
′
S 
′
  using a policy derived from 
Q
Q (e.g., greedy)
Update the action-value: 
Q
(
S
,
A
)
←
Q
(
S
,
A
)
+
α
[
R
+
γ
Q
(
S
′
,
A
′
)
−
Q
(
S
,
A
)
]
Q(S,A)←Q(S,A)+α[R+γQ(S 
′
 ,A 
′
 )−Q(S,A)]
Set 
S
←
S
′
S←S 
′
 , 
A
←
A
′
A←A 
′
 
Repeat until 
S
S is terminal
This algorithm continuously improves the policy as it explores and learns from interaction, making it suitable for online reinforcement learning scenarios.

"""
from collections import defaultdict

def sarsa_update(transitions, initial_states, alpha, gamma, max_steps):
    Q = defaultdict(float)

    # Ensure all Q-values are initialized
    for (state, action) in transitions:
        Q[(state, action)] = 0.0

    def get_possible_actions(state):
        return [a for (s, a) in transitions if s == state]

    def greedy_action(state):
        actions = get_possible_actions(state)
        if not actions:
            return None
        max_q = max(Q[(state, a)] for a in actions)
        best_actions = [a for a in actions if Q[(state, a)] == max_q]
        return best_actions[0]  # deterministic tie-break

    for init_state in initial_states:
        state = init_state
        action = greedy_action(state)
        steps = 0

        while state != 'terminal' and steps < max_steps and action is not None:
            reward, next_state = transitions[(state, action)]
            next_action = greedy_action(next_state)

            q_current = Q[(state, action)]
            q_next = Q[(next_state, next_action)] if next_action else 0.0
            Q[(state, action)] += alpha * (reward + gamma * q_next - q_current)

            state = next_state
            action = next_action
            steps += 1

    return dict(Q)
