"""
Positional Encoding Calculator

Write a Python function to implement the Positional Encoding layer for Transformers. The function should calculate positional encodings for a sequence length (position) and model dimensionality (d_model) using sine and cosine functions as specified in the Transformer architecture. The function should return -1 if position is 0, or if d_model is less than or equal to 0. The output should be a numpy array of type float16.

Example:
Input:
position = 2, d_model = 8
Output:
[[[ 0.,0.,0.,0.,1.,1.,1.,1.,]
  [ 0.8413,0.0998,0.01,0.001,0.5405,0.995,1.,1.]]]
Reasoning:
The function computes the positional encoding by calculating sine values for even indices and cosine values for odd indices, ensuring that the encoding provides the required positional information.

"""
"""
The Positional Encoding Layer in Transformers
The Positional Encoding layer in Transformers plays a critical role by providing necessary positional information to the model. This is particularly important because the Transformer architecture, unlike RNNs or LSTMs, processes input sequences in parallel and lacks inherent mechanisms to account for the sequential order of tokens.

The mathematical intuition behind the Positional Encoding layer in Transformers is centered on enabling the model to incorporate information about the order of tokens in a sequence.

Function Parameters
position: Total positions or length of the sequence.
d_model: Dimensionality of the model's output.
Generating the Base Matrix
angle_rads: Creates a matrix where rows represent sequence positions and columns represent feature dimensions. Values are scaled by dividing each position index by:
10000
2
⋅
i
d
m
o
d
e
l
10000 
d 
model
​
 
2⋅i
​
 
 
Applying Sine and Cosine Functions
For even indices: Apply the sine function to encode positions.
P
E
(
pos
,
2
i
)
=
sin
⁡
(
pos
10000
2
i
d
m
o
d
e
l
)
PE(pos,2i)=sin( 
10000 
d 
model
​
 
2i
​
 
 
pos
​
 )

For odd indices: Apply the cosine function for a phase-shifted encoding.
P
E
(
pos
,
2
i
+
1
)
=
cos
⁡
(
pos
10000
2
i
d
m
o
d
e
l
)
PE(pos,2i+1)=cos( 
10000 
d 
model
​
 
2i
​
 
 
pos
​
 )

Creating the Positional Encoding Tensor
The matrix is expanded to match input shape expectations of models like Transformers and cast to float32.
Output
Returns a TensorFlow tensor of shape 
(
1
,
position
,
d_model
)
(1,position,d_model), ready to be added to input embeddings to incorporate positional information.
"""

import numpy as np

def pos_encoding(position: int, d_model: int):
    
    if position == 0 or d_model <= 0:
        return -1

    # Create position and dimension indices
    pos = np.arange(position, dtype=np.float32).reshape(position, 1)
    ind = np.arange(d_model, dtype=np.float32).reshape(1, d_model)

    # Compute the angles
    angle_rads = pos / np.power(10000, (2 * (ind // 2)) / d_model)

    # Apply sine to even indices, cosine to odd indices
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])  # Even indices (0, 2, 4...)
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])  # Odd indices (1, 3, 5...)

    # Convert to float16 as required
    return angle_rads.astype(np.float16)
