"""
Task: Implement Long Short-Term Memory (LSTM) Network
Your task is to implement an LSTM network that processes a sequence of inputs and produces the final hidden state and cell state after processing all inputs.

Write a class LSTM with the following methods:

__init__(self, input_size, hidden_size): Initializes the LSTM with random weights and zero biases.
forward(self, x, initial_hidden_state, initial_cell_state): Processes a sequence of inputs and returns the hidden states at each time step, as well as the final hidden state and cell state.
The LSTM should compute the forget gate, input gate, candidate cell state, and output gate at each time step to update the hidden state and cell state.

Example:
Input:
input_sequence = np.array([[1.0], [2.0], [3.0]])
initial_hidden_state = np.zeros((1, 1))
initial_cell_state = np.zeros((1, 1))

lstm = LSTM(input_size=1, hidden_size=1)
outputs, final_h, final_c = lstm.forward(input_sequence, initial_hidden_state, initial_cell_state)

print(final_h)
Output:
[[0.73698596]] (approximate)
Reasoning:
The LSTM processes the input sequence [1.0, 2.0, 3.0] and produces the final hidden state [0.73698596].

"""
"""
Understanding Long Short-Term Memory Networks (LSTMs)
Long Short-Term Memory Networks are a special type of RNN designed to capture long-term dependencies in sequential data by using a more complex hidden state structure.

LSTM Gates and Their Functions
For each time step 
t
t, the LSTM updates its cell state 
c
t
c 
t
​
  and hidden state 
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
 , the previous cell state 
c
t
−
1
c 
t−1
​
 , and the previous hidden state 
h
t
−
1
h 
t−1
​
 . The LSTM architecture consists of several gates that control the flow of information:

Forget Gate 
f
t
f 
t
​
 :
This gate decides what information to discard from the cell state. It looks at the previous hidden state 
h
t
−
1
h 
t−1
​
  and the current input 
x
t
x 
t
​
 , and outputs a number between 0 and 1 for each number in the cell state. A 1 represents "keep this" while a 0 represents "forget this".

f
t
=
σ
(
W
f
⋅
[
h
t
−
1
,
x
t
]
+
b
f
)
f 
t
​
 =σ(W 
f
​
 ⋅[h 
t−1
​
 ,x 
t
​
 ]+b 
f
​
 )
Input Gate 
i
t
i 
t
​
 :
This gate decides which new information will be stored in the cell state. It consists of two parts:

A sigmoid layer that decides which values we'll update.
A tanh layer that creates a vector of new candidate values 
c
~
t
c
~
  
t
​
  that could be added to the state.
i
t
=
σ
(
W
i
⋅
[
h
t
−
1
,
x
t
]
+
b
i
)
i 
t
​
 =σ(W 
i
​
 ⋅[h 
t−1
​
 ,x 
t
​
 ]+b 
i
​
 )
c
~
t
=
tanh
⁡
(
W
c
⋅
[
h
t
−
1
,
x
t
]
+
b
c
)
c
~
  
t
​
 =tanh(W 
c
​
 ⋅[h 
t−1
​
 ,x 
t
​
 ]+b 
c
​
 )
Cell State Update 
c
t
c 
t
​
 :
This step updates the old cell state 
c
t
−
1
c 
t−1
​
  into the new cell state 
c
t
c 
t
​
 . It multiplies the old state by the forget gate output, then adds the product of the input gate and the new candidate values.

c
t
=
f
t
∘
c
t
−
1
+
i
t
∘
c
~
t
c 
t
​
 =f 
t
​
 ∘c 
t−1
​
 +i 
t
​
 ∘ 
c
~
  
t
​
 
Output Gate 
o
t
o 
t
​
 :
This gate decides what parts of the cell state we're going to output. It uses a sigmoid function to determine which parts of the cell state to output, and then multiplies it by a tanh of the cell state to get the final output.

o
t
=
σ
(
W
o
⋅
[
h
t
−
1
,
x
t
]
+
b
o
)
o 
t
​
 =σ(W 
o
​
 ⋅[h 
t−1
​
 ,x 
t
​
 ]+b 
o
​
 )
h
t
=
o
t
∘
tanh
⁡
(
c
t
)
h 
t
​
 =o 
t
​
 ∘tanh(c 
t
​
 )
Where:

(
W
f
,
W
i
,
W
c
,
W
o
)
(W 
f
​
 ,W 
i
​
 ,W 
c
​
 ,W 
o
​
 ) are weight matrices for the forget gate, input gate, cell state, and output gate respectively.
(
b
f
,
b
i
,
b
c
,
b
o
)
(b 
f
​
 ,b 
i
​
 ,b 
c
​
 ,b 
o
​
 ) are bias vectors.
σ
σ is the sigmoid activation function.
∘
∘ denotes element-wise multiplication.
Implementation Steps
Initialization: Start with the initial cell state 
c
0
c 
0
​
  and hidden state 
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
Compute forget gate 
f
t
f 
t
​
 , input gate 
i
t
i 
t
​
 , candidate cell state 
c
~
t
c
~
  
t
​
 , and output gate 
o
t
o 
t
​
 .
Update cell state 
c
t
c 
t
​
  and hidden state 
h
t
h 
t
​
 .
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
Initial states: 
c
0
=
0.0
c 
0
​
 =0.0, 
h
0
=
0.0
h 
0
​
 =0.0
Simplified weights (for demonstration): 
W
f
=
W
i
=
W
c
=
W
o
=
0.5
W 
f
​
 =W 
i
​
 =W 
c
​
 =W 
o
​
 =0.5
All biases: 
b
f
=
b
i
=
b
c
=
b
o
=
0.1
b 
f
​
 =b 
i
​
 =b 
c
​
 =b 
o
​
 =0.1
Compute:
First time step 
t
=
1
t=1:

f
1
=
σ
(
0.5
×
1.0
+
0.1
)
=
0.6487
f 
1
​
 =σ(0.5×1.0+0.1)=0.6487
i
1
=
σ
(
0.5
×
1.0
+
0.1
)
=
0.6487
i 
1
​
 =σ(0.5×1.0+0.1)=0.6487
c
~
1
=
tanh
⁡
(
0.5
×
1.0
+
0.1
)
=
0.5370
c
~
  
1
​
 =tanh(0.5×1.0+0.1)=0.5370
c
1
=
f
1
×
0.0
+
i
1
×
c
~
1
=
0.6487
×
0.0
+
0.6487
×
0.5370
=
0.3484
c 
1
​
 =f 
1
​
 ×0.0+i 
1
​
 × 
c
~
  
1
​
 =0.6487×0.0+0.6487×0.5370=0.3484
o
1
=
σ
(
0.5
×
1.0
+
0.1
)
=
0.6487
o 
1
​
 =σ(0.5×1.0+0.1)=0.6487
h
1
=
o
1
×
tanh
⁡
(
c
1
)
=
0.6487
×
tanh
⁡
(
0.3484
)
=
0.2169
h 
1
​
 =o 
1
​
 ×tanh(c 
1
​
 )=0.6487×tanh(0.3484)=0.2169
Second time step 
t
=
2
t=2: (Calculations omitted for brevity, but follow the same pattern using 
x
2
=
2.0
x 
2
​
 =2.0 and the previous states)

Third time step 
t
=
3
t=3: (Calculations omitted for brevity, but follow the same pattern using 
x
3
=
3.0
x 
3
​
 =3.0 and the previous states)

The final hidden state 
h
3
h 
3
​
  would be the result after these calculations.

Applications
LSTMs are extensively used in various sequence modeling tasks, including machine translation, speech recognition, and time series forecasting, where capturing long-term dependencies is crucial.
"""
import numpy as np

class LSTM:
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Initialize weights and biases
        self.Wf = np.random.randn(hidden_size, input_size + hidden_size)
        self.Wi = np.random.randn(hidden_size, input_size + hidden_size)
        self.Wc = np.random.randn(hidden_size, input_size + hidden_size)
        self.Wo = np.random.randn(hidden_size, input_size + hidden_size)

        self.bf = np.zeros((hidden_size, 1))
        self.bi = np.zeros((hidden_size, 1))
        self.bc = np.zeros((hidden_size, 1))
        self.bo = np.zeros((hidden_size, 1))

    def forward(self, x, initial_hidden_state, initial_cell_state):
        h = initial_hidden_state
        c = initial_cell_state
        outputs = []

        for t in range(len(x)):
            xt = x[t].reshape(-1, 1)
            concat = np.vstack((h, xt))

            # Forget gate
            ft = self.sigmoid(np.dot(self.Wf, concat) + self.bf)

            # Input gate
            it = self.sigmoid(np.dot(self.Wi, concat) + self.bi)
            c_tilde = np.tanh(np.dot(self.Wc, concat) + self.bc)

            # Cell state update
            c = ft * c + it * c_tilde

            # Output gate
            ot = self.sigmoid(np.dot(self.Wo, concat) + self.bo)

            # Hidden state update
            h = ot * np.tanh(c)

            outputs.append(h)

        return np.array(outputs), h, c

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
