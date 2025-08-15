"""
Implement a function to simulate a Markov Chain. The function should take a 2D numpy array representing the transition matrix (where each row sums to 1), an integer for the initial state index, and an integer for the number of steps to simulate. It should return a numpy array containing the sequence of state indices over time, including the initial state. Use numpy for array operations and random selections.

Example:
Input:
transition_matrix = np.array([[0.8, 0.2], [0.3, 0.7]]); print(simulate_markov_chain(transition_matrix, 0, 3))
Output:
[0 0 1 1]
Reasoning:
The solution simulates a Markov chain by starting with the initial state (0) and iteratively selecting the next state based on the probabilities in the transition matrix. For the given input, this process generates the sequence [0, 0, 1, 1] over three steps, where the first state is the initial one, and subsequent states are chosen such that from state 0, it stays at 0, then transitions to 1, and remains at 1.


Markov Chains: A Stochastic Process
Markov Chains are a fundamental concept in probability theory, used to model systems that transition between different states over time. In this explanation, we will explore the key ideas behind Markov Chains, focusing on their mathematical foundations and intuitive meaning.

1. Definition of a Markov Chain
A Markov Chain is a sequence of events or states where the probability of moving to the next state depends only on the current state, not on any previous states. This property is known as the "memoryless" or Markov property.

To illustrate, imagine a system with a set of possible states, such as weather conditions (e.g., sunny or rainy). At any given moment, the system occupies one state, and the likelihood of transitioning to another state is determined solely by the current one. Mathematically, if we denote the states as 
S
1
,
S
2
,
…
,
S
n
S 
1
​
 ,S 
2
​
 ,…,S 
n
​
 , the process evolves according to probabilities that satisfy the Markov property.

For example, the equation for the probability of being in state 
S
j
S 
j
​
  at time 
t
+
1
t+1, given the current state 
S
i
S 
i
​
  at time 
t
t, can be expressed as:

P
(
S
t
+
1
=
S
j
∣
S
t
=
S
i
,
S
t
−
1
,
…
 
)
=
P
(
S
t
+
1
=
S
j
∣
S
t
=
S
i
)
P(S 
t+1
​
 =S 
j
​
 ∣S 
t
​
 =S 
i
​
 ,S 
t−1
​
 ,…)=P(S 
t+1
​
 =S 
j
​
 ∣S 
t
​
 =S 
i
​
 )

Here, 
P
P represents probability, 
S
t
S 
t
​
  is the state at time 
t
t, and the right side shows that only the current state 
S
t
S 
t
​
  matters. This equation highlights how the process simplifies decision-making by ignoring historical data, making it useful for modeling random phenomena like random walks or population dynamics.

2. Transition Probabilities and the Matrix
At the heart of a Markov Chain is the concept of transition probabilities, which quantify the likelihood of moving from one state to another. These probabilities are organized into a structure called a transition matrix.

A transition matrix is a square array where each entry represents the probability of transitioning from a specific row state to a specific column state. For a system with 
n
n states, the matrix is an 
n
×
n
n×n grid, and each row sums to 1, ensuring that the probabilities for all possible outcomes from a given state add up to certainty.

The general form of a transition matrix 
P
P is:

P
=
(
p
11
p
12
⋯
p
1
n
p
21
p
22
⋯
p
2
n
⋮
⋮
⋱
⋮
p
n
1
p
n
2
⋯
p
n
n
)
P= 
​
  
p 
11
​
 
p 
21
​
 
⋮
p 
n1
​
 
​
  
p 
12
​
 
p 
22
​
 
⋮
p 
n2
​
 
​
  
⋯
⋯
⋱
⋯
​
  
p 
1n
​
 
p 
2n
​
 
⋮
p 
nn
​
 
​
  
​
 

In this matrix, 
p
i
j
p 
ij
​
  is the probability of transitioning from state 
i
i to state 
j
j. For instance, 
p
11
p 
11
​
  might represent the probability of staying in state 1, while 
p
12
p 
12
​
  represents the probability of moving from state 1 to state 2. Each 
p
i
j
p 
ij
​
  is a value between 0 and 1, and the sum of each row equals 1, as expressed by:

∑
j
=
1
n
p
i
j
=
1
for each row 
i
∑ 
j=1
n
​
 p 
ij
​
 =1for each row i

This equation ensures that the matrix reflects a complete set of possibilities for each starting state, providing a clear framework for predicting future behavior based on the current position.

3. Evolution of States Over Time
Once the transition matrix is defined, we can describe how the states of a Markov Chain evolve through successive steps. Starting from an initial state, the process generates a sequence of states by applying the transition probabilities repeatedly.

At each step, the next state is determined by the probabilities associated with the current state. Over multiple steps, this leads to a sequence that can be analyzed to understand long-term patterns, such as whether the system tends to settle into certain states or remains unpredictable.

Mathematically, if we begin in state 
i
i at time 0, the probability of being in state 
j
j after one step is given by the entry 
p
i
j
p 
ij
​
  in the matrix. After multiple steps, the overall probability distribution can be computed by multiplying the initial state probabilities by the transition matrix raised to the power of the number of steps. For a probability vector 
v
t
v 
t
​
  representing the likelihood of being in each state at time 
t
t, the evolution is:

v
t
+
1
=
v
t
⋅
P
v 
t+1
​
 =v 
t
​
 ⋅P

Here, 
v
t
v 
t
​
  is a row vector of probabilities summing to 1, and 
P
P is the transition matrix. This operation shows how the distribution shifts over time, with each multiplication reflecting the application of transition rules. In the long run, many Markov Chains reach a steady-state distribution, where the probabilities no longer change, offering insights into stable behaviors of the system.

Example Walkthrough
To make the concept more concrete, consider a simple two-state system modeling weather patterns: State 1 as "Sunny" and State 2 as "Rainy." Suppose the transition matrix is:

P
=
(
0.7
0.3
0.4
0.6
)
P=( 
0.7
0.4
​
  
0.3
0.6
​
 )

In this matrix, the entry 0.7 means there is a 70% chance of staying Sunny if it is currently Sunny, while 0.3 means a 30% chance of becoming Rainy. Similarly, 0.4 indicates a 40% chance of becoming Sunny if it is currently Rainy, and 0.6 means a 60% chance of staying Rainy.

Starting from State 1 (Sunny), after one step, there is a 70% chance of remaining Sunny and a 30% chance of moving to Rainy. If it becomes Rainy, the next step would follow the second row of the matrix. Over several steps, this process might fluctuate, but eventually, it could approach a balance where the probabilities stabilize, reflecting typical weather patterns in this model. This example demonstrates how transition probabilities guide the system's behavior in a predictable yet random manner.
"""
import numpy as np
def simulate_markov_chain(transition_matrix, initial_state, num_steps):
    # Your code here
    states = [initial_state]
    current_state = initial_state 
    for _ in range(num_steps):
        next_state = np.random.choice(
            a = len(transition_matrix),
            p = transition_matrix[current_state]
        )
        states.append(next_state)
        current_state = next_state 
    return np.array(states)
