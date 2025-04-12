"""
Write a Python function that implements a simple Recurrent Neural Network (RNN) cell. The function should process a sequence of input vectors and produce the final hidden state. Use the tanh activation function for the hidden state updates. The function should take as inputs the sequence of input vectors, the initial hidden state, the weight matrices for input-to-hidden and hidden-to-hidden connections, and the bias vector. The function should return the final hidden state after processing the entire sequence, rounded to four decimal places.

Example:
Input:
input_sequence = [[1.0], [2.0], [3.0]]
    initial_hidden_state = [0.0]
    Wx = [[0.5]]  # Input to hidden weights
    Wh = [[0.8]]  # Hidden to hidden weights
    b = [0.0]     # Bias
Output:
final_hidden_state = [0.9993]

"""
"""
Understanding Recurrent Neural Networks (RNNs)
Recurrent Neural Networks (RNNs) are a class of neural networks designed to handle sequential data by maintaining a hidden state that captures information from previous inputs.

Mathematical Formulation
For each time step 
t
t, the RNN updates its hidden state 
h
t
h 
t
​
  using the current input 
x
t
x 
t
​
  and the previous hidden state 
h
t
−
1
h 
t−1
​
 :

h
t
=
tanh
⁡
(
W
x
x
t
+
W
h
h
t
−
1
+
b
)
h 
t
​
 =tanh(W 
x
​
 x 
t
​
 +W 
h
​
 h 
t−1
​
 +b)
Where:

W
x
W 
x
​
  is the weight matrix for the input-to-hidden connections.
W
h
W 
h
​
  is the weight matrix for the hidden-to-hidden connections.
b
b is the bias vector.
tanh
⁡
tanh is the hyperbolic tangent activation function applied element-wise.
Implementation Steps
Initialization: Start with the initial hidden state 
h
0
h 
0
​
 .

Sequence Processing: For each input 
x
t
x 
t
​
  in the sequence:

h
t
=
tanh
⁡
(
W
x
x
t
+
W
h
h
t
−
1
+
b
)
h 
t
​
 =tanh(W 
x
​
 x 
t
​
 +W 
h
​
 h 
t−1
​
 +b)
Final Output: After processing all inputs, the final hidden state 
h
T
h 
T
​
  (where 
T
T is the length of the sequence) contains information from the entire sequence.

Example Calculation
Given:

Inputs: 
x
1
=
1.0
x 
1
​
 =1.0, 
x
2
=
2.0
x 
2
​
 =2.0, 
x
3
=
3.0
x 
3
​
 =3.0
Initial hidden state: 
h
0
=
0.0
h 
0
​
 =0.0
Weights:
W
x
=
0.5
W 
x
​
 =0.5
W
h
=
0.8
W 
h
​
 =0.8
Bias: 
b
=
0.0
b=0.0
Compute:

First time step (
t
=
1
t=1):

h
1
=
tanh
⁡
(
0.5
×
1.0
+
0.8
×
0.0
+
0.0
)
=
tanh
⁡
(
0.5
)
≈
0.4621
h 
1
​
 =tanh(0.5×1.0+0.8×0.0+0.0)=tanh(0.5)≈0.4621
Second time step (
t
=
2
t=2):

h
2
=
tanh
⁡
(
0.5
×
2.0
+
0.8
×
0.4621
+
0.0
)
=
tanh
⁡
(
1.0
+
0.3697
)
=
tanh
⁡
(
1.3697
)
≈
0.8781
h 
2
​
 =tanh(0.5×2.0+0.8×0.4621+0.0)=tanh(1.0+0.3697)=tanh(1.3697)≈0.8781
Third time step (
t
=
3
t=3):

h
3
=
tanh
⁡
(
0.5
×
3.0
+
0.8
×
0.8781
+
0.0
)
=
tanh
⁡
(
1.5
+
0.7025
)
=
tanh
⁡
(
2.2025
)
≈
0.9750
h 
3
​
 =tanh(0.5×3.0+0.8×0.8781+0.0)=tanh(1.5+0.7025)=tanh(2.2025)≈0.9750
The final hidden state 
h
3
h 
3
​
  is approximately 0.9750.

Applications
RNNs are widely used in natural language processing, time-series prediction, and any task involving sequential data.

"""

import numpy as np

def rnn_forward(input_sequence, initial_hidden_state, Wx, Wh, b):
    h = np.array(initial_hidden_state)
    Wx = np.array(Wx)
    Wh = np.array(Wh)
    b = np.array(b)
    for x in input_sequence:
        x = np.array(x)
        h = np.tanh(np.dot(Wx, x) + np.dot(Wh, h) + b)
    final_hidden_state = np.round(h, 4)
    return final_hidden_state.tolist()
