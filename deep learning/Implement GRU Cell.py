"""
Problem
Implement a Gated Recurrent Unit (GRU) cell forward pass. The GRU is a type of recurrent neural network architecture that uses gating mechanisms to control the flow of information, helping to mitigate the vanishing gradient problem.

A GRU cell computes a new hidden state given an input vector and the previous hidden state using update and reset gates.

Input Parameters:
x: Input vector of shape (input_size,)
h_prev: Previous hidden state of shape (hidden_size,)
W_z, W_r, W_h: Weight matrices for input of shape (hidden_size, input_size)
U_z, U_r, U_h: Weight matrices for hidden state of shape (hidden_size, hidden_size)
b_z, b_r, b_h: Bias vectors of shape (hidden_size,)
Output:
h_next: New hidden state of shape (hidden_size,)
The GRU uses sigmoid and tanh activation functions. The update gate controls how much of the previous hidden state to retain, while the reset gate controls how much of the previous hidden state to forget when computing the candidate hidden state.

Example:
Input:
x = [1.0, 0.5], h_prev = [0.0, 0.0, 0.0], all weight matrices filled with 0.1 or 0.2, all biases are zeros
Output:
[0.1565, 0.1565, 0.1565]
Reasoning:
With h_prev = 0, the reset gate has no effect. The update gate z = sigmoid(0.15) = 0.5374 for each unit. The candidate h_tilde = tanh(0.3) = 0.2913 for each unit. The final hidden state h_next = z * h_tilde = 0.5374 * 0.2913 = 0.1565 for each unit.

Learn About topic
Gated Recurrent Unit (GRU)
The GRU is a gating mechanism in recurrent neural networks introduced by Cho et al. (2014). It is designed to solve the vanishing gradient problem that affects standard RNNs.

Architecture
A GRU cell uses two gates to control information flow:

Update Gate (
z
t
z 
t
​
 ): Controls how much of the previous hidden state to retain.
Reset Gate (
r
t
r 
t
​
 ): Controls how much of the previous hidden state to use when computing the candidate hidden state.
Forward Pass Equations
Given input 
x
t
x 
t
​
  and previous hidden state 
h
t
−
1
h 
t−1
​
 , the GRU computes:

Update Gate: 
z
t
=
σ
(
W
z
x
t
+
U
z
h
t
−
1
+
b
z
)
z 
t
​
 =σ(W 
z
​
 x 
t
​
 +U 
z
​
 h 
t−1
​
 +b 
z
​
 )

Reset Gate: 
r
t
=
σ
(
W
r
x
t
+
U
r
h
t
−
1
+
b
r
)
r 
t
​
 =σ(W 
r
​
 x 
t
​
 +U 
r
​
 h 
t−1
​
 +b 
r
​
 )

Candidate Hidden State: 
h
~
t
=
tanh
⁡
(
W
h
x
t
+
U
h
(
r
t
⊙
h
t
−
1
)
+
b
h
)
h
~
  
t
​
 =tanh(W 
h
​
 x 
t
​
 +U 
h
​
 (r 
t
​
 ⊙h 
t−1
​
 )+b 
h
​
 )

New Hidden State: 
h
t
=
(
1
−
z
t
)
⊙
h
t
−
1
+
z
t
⊙
h
~
t
h 
t
​
 =(1−z 
t
​
 )⊙h 
t−1
​
 +z 
t
​
 ⊙ 
h
~
  
t
​
 

Where:

σ
σ is the sigmoid function: 
σ
(
x
)
=
1
1
+
e
−
x
σ(x)= 
1+e 
−x
 
1
​
 
⊙
⊙ denotes element-wise multiplication
W
z
,
W
r
,
W
h
W 
z
​
 ,W 
r
​
 ,W 
h
​
  are input weight matrices of shape 
(
hidden_size
,
input_size
)
(hidden_size,input_size)
U
z
,
U
r
,
U
h
U 
z
​
 ,U 
r
​
 ,U 
h
​
  are recurrent weight matrices of shape 
(
hidden_size
,
hidden_size
)
(hidden_size,hidden_size)
b
z
,
b
r
,
b
h
b 
z
​
 ,b 
r
​
 ,b 
h
​
  are bias vectors of shape 
(
hidden_size
,
)
(hidden_size,)
Intuition
When 
z
t
≈
0
z 
t
​
 ≈0: The new hidden state is mostly the previous state 
h
t
−
1
h 
t−1
​
 
When 
z
t
≈
1
z 
t
​
 ≈1: The new hidden state is mostly the candidate 
h
~
t
h
~
  
t
​
 
When 
r
t
≈
0
r 
t
​
 ≈0: The candidate ignores the previous hidden state
When 
r
t
≈
1
r 
t
​
 ≈1: The candidate fully considers the previous hidden state
Comparison with LSTM
The GRU is a simplified version of the LSTM with fewer parameters:

GRU has 2 gates vs. LSTM's 3 gates
GRU merges the cell state and hidden state
GRU typically trains faster while achieving comparable performance

"""

import numpy as np

def gru_cell(x: np.ndarray, h_prev: np.ndarray,
             W_z: np.ndarray, U_z: np.ndarray, b_z: np.ndarray,
             W_r: np.ndarray, U_r: np.ndarray, b_r: np.ndarray,
             W_h: np.ndarray, U_h: np.ndarray, b_h: np.ndarray) -> np.ndarray:
    """
    Implements a single GRU cell forward pass.
    
    Args:
        x: Input vector of shape (input_size,)
        h_prev: Previous hidden state of shape (hidden_size,)
        W_z, W_r, W_h: Weight matrices for input of shape (hidden_size, input_size)
        U_z, U_r, U_h: Weight matrices for hidden state of shape (hidden_size, hidden_size)
        b_z, b_r, b_h: Bias vectors of shape (hidden_size,)
    
    Returns:
        h_next: New hidden state of shape (hidden_size,)
    """
    # Sigmoid activation function
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))
    
    # 1. Compute update gate
    # z_t = σ(W_z · x + U_z · h_prev + b_z)
    z_t = sigmoid(np.dot(W_z, x) + np.dot(U_z, h_prev) + b_z)
    
    # 2. Compute reset gate
    # r_t = σ(W_r · x + U_r · h_prev + b_r)
    r_t = sigmoid(np.dot(W_r, x) + np.dot(U_r, h_prev) + b_r)
    
    # 3. Compute candidate hidden state
    # h_tilde = tanh(W_h · x + U_h · (r_t ⊙ h_prev) + b_h)
    # Element-wise multiplication: r_t ⊙ h_prev
    h_tilde = np.tanh(np.dot(W_h, x) + np.dot(U_h, r_t * h_prev) + b_h)
    
    # 4. Compute new hidden state
    # h_next = (1 - z_t) ⊙ h_prev + z_t ⊙ h_tilde
    h_next = (1 - z_t) * h_prev + z_t * h_tilde
    
    return h_next
