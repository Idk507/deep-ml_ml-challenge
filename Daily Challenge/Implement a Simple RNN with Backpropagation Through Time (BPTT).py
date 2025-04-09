"""
Implement a Simple RNN with Backpropagation Through Time (BPTT)

Task: Implement a Simple RNN with Backpropagation Through Time (BPTT)
Your task is to implement a simple Recurrent Neural Network (RNN) and backpropagation through time (BPTT) to learn from sequential data. The RNN will process input sequences, update hidden states, and perform backpropagation to adjust weights based on the error gradient.

Write a class SimpleRNN with the following methods:

__init__(self, input_size, hidden_size, output_size): Initializes the RNN with random weights and zero biases.
forward(self, x): Processes a sequence of inputs and returns the hidden states and output.
backward(self, x, y, learning_rate): Performs backpropagation through time (BPTT) to adjust the weights based on the loss.
In this task, the RNN will be trained on sequence prediction, where the network will learn to predict the next item in a sequence. You should use 1/2 * Mean Squared Error (MSE) as the loss function and make sure to aggregate the losses at each time step by summing.

Example:
Input:
import numpy as np
    input_sequence = np.array([[1.0], [2.0], [3.0], [4.0]])
    expected_output = np.array([[2.0], [3.0], [4.0], [5.0]])
    # Initialize RNN
    rnn = SimpleRNN(input_size=1, hidden_size=5, output_size=1)
    
    # Forward pass
    output = rnn.forward(input_sequence)
    
    # Backward pass
    rnn.backward(input_sequence, expected_output, learning_rate=0.01)
    
    print(output)
    
    # The output should show the RNN predictions for each step of the input sequence.
Output:
[[x1], [x2], [x3], [x4]]
Reasoning:
The RNN processes the input sequence [1.0, 2.0, 3.0, 4.0] and predicts the next item in the sequence at each step
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

class SimpleRNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.hidden_size = hidden_size
        self.W_xh = np.random.randn(hidden_size, input_size) * 0.01
        self.W_hh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.W_hy = np.random.randn(output_size, hidden_size) * 0.01
        self.b_h = np.zeros((hidden_size, 1))
        self.b_y = np.zeros((output_size, 1))

    def forward(self, x):
        h = np.zeros((self.hidden_size, 1))  # Initialize hidden state
        outputs = []
        self.last_inputs = []
        self.last_hiddens = [h]
        
        for t in range(len(x)):
            self.last_inputs.append(x[t].reshape(-1, 1))
            h = np.tanh(np.dot(self.W_xh, self.last_inputs[t]) + np.dot(self.W_hh, h) + self.b_h)
            y = np.dot(self.W_hy, h) + self.b_y
            outputs.append(y)
            self.last_hiddens.append(h)
        
        self.last_outputs = outputs
        return np.array(outputs)

    def backward(self, x, y, learning_rate):
        dW_xh = np.zeros_like(self.W_xh)
        dW_hh = np.zeros_like(self.W_hh)
        dW_hy = np.zeros_like(self.W_hy)
        db_h = np.zeros_like(self.b_h)
        db_y = np.zeros_like(self.b_y)

        dh_next = np.zeros((self.hidden_size, 1))

        for t in reversed(range(len(x))):
            dy = self.last_outputs[t] - y[t].reshape(-1, 1)  # (Predicted - Actual)
            dW_hy += np.dot(dy, self.last_hiddens[t+1].T)
            db_y += dy

            dh = np.dot(self.W_hy.T, dy) + dh_next
            dh_raw = (1 - self.last_hiddens[t+1] ** 2) * dh  # Derivative of tanh

            dW_xh += np.dot(dh_raw, self.last_inputs[t].T)
            dW_hh += np.dot(dh_raw, self.last_hiddens[t].T)
            db_h += dh_raw

            dh_next = np.dot(self.W_hh.T, dh_raw)

        # Update weights and biases
        self.W_xh -= learning_rate * dW_xh
        self.W_hh -= learning_rate * dW_hh
        self.W_hy -= learning_rate * dW_hy
        self.b_h -= learning_rate * db_h
        self.b_y -= learning_rate * db_y
