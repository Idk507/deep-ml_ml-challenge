"""
Create a function dense_net_block that performs the forward pass of a DenseNet dense block on a batch of images stored in an NHWC NumPy tensor input_data (shape (N, H, W, C0)). The block must run num_layers iterations; at each iteration it should (i) apply ReLU to the running feature tensor, (ii) convolve it with the corresponding kernel from kernels (using stride 1, no bias, and symmetric zero-padding so that H and W are preserved), and (iii) concatenate the convolution output (whose channel count equals growth_rate) to the running tensor along the channel axis. Every kernel kernels[l] therefore has shape (kh, kw, C0 + l x growth_rate, growth_rate), where (kh, kw) equals kernel_size (default (3, 3)). After the final layer the function must return a tensor of shape (N, H, W, C0 + num_layers x growth_rate). If any kernel's input-channel dimension does not match the current feature-map channels, the function should raise a ValueError.

Example:
Input:
X = np.random.randn(1, 2, 2, 1); kernels = [np.random.randn(3, 3, 2 + i*1, 1) * 0.01 for i in range(2)]; print(dense_net_block(X, 2, 1, kernels))
Output:
[[[[ 4.96714153e-01, -1.38264301e-01, -2.30186127e-03, -6.70426255e-05]]]]
Reasoning:
Each dense block layer concatenates its output to the existing feature maps, expanding the number of output channels by 1 per layer (the growth rate). After 2 layers, the original 2 channels become 4.
Understanding Dense Blocks and 2D Convolutions
Dense blocks are a key innovation in the DenseNet architecture. Each layer receives input from all previous layers, leading to rich feature reuse and efficient gradient flow.

Dense Block Concept
For a dense block:

Each layer: Applies ReLU, then 2D convolution, and then concatenates the output to previous features.
Mathematically:
x
l
=
H
l
(
[
x
0
,
x
1
,
…
,
x
l
−
1
]
)
x 
l
​
 =H 
l
​
 ([x 
0
​
 ,x 
1
​
 ,…,x 
l−1
​
 ])
where 
H
l
(
⋅
)
H 
l
​
 (⋅) is the convolution and activation operations.

2D Convolution Basics
A 2D convolution at a position 
(
i
,
j
)
(i,j) for input 
X
X and kernel 
K
K is:

Y
[
i
,
j
]
=
∑
m
=
0
k
h
−
1
∑
n
=
0
k
w
−
1
X
[
i
+
m
,
j
+
n
]
⋅
K
[
m
,
n
]
Y[i,j]= 
m=0
∑
k 
h
​
 −1
​
  
n=0
∑
k 
w
​
 −1
​
 X[i+m,j+n]⋅K[m,n]
Padding to Preserve Spatial Dimensions
To preserve height and width:

padding
=
k
−
1
2
padding= 
2
k−1
​
 
Dense Block Growth
Each layer adds 
growth rate
growth rate channels.
After 
L
L layers, total channels = input channels + 
L
×
growth rate
L×growth rate.
Putting It All Together
1ï¸â£ Start with an input tensor.
2ï¸â£ Repeat for 
num layers
num layers:

Apply ReLU activation.
Apply 2D convolution (with padding).
Concatenate the output along the channel dimension.
By understanding these core principles, youâre ready to build the dense block function!
"""
import numpy as np

def dense_net_block(input_data, num_layers, growth_rate, kernels, kernel_size=(3, 3)):
    N, H, W, C0 = input_data.shape
    kh, kw = kernel_size
    padding_h = kh // 2
    padding_w = kw // 2

    # Initialize the feature tensor
    features = input_data

    for l in range(num_layers):
        # Apply ReLU activation
        activated = np.maximum(features, 0)

        # Validate kernel dimensions
        expected_in_channels = C0 + l * growth_rate
        kernel = kernels[l]
        if kernel.shape != (kh, kw, expected_in_channels, growth_rate):
            raise ValueError(
                f"Layer {l}: Expected kernel shape {(kh, kw, expected_in_channels, growth_rate)}, "
                f"got {kernel.shape}"
            )

        # Pad the input symmetrically (NHWC)
        padded = np.pad(activated, 
                        ((0, 0), (padding_h, padding_h), (padding_w, padding_w), (0, 0)), 
                        mode='constant')

        # Perform convolution for each example in the batch
        conv_out = np.zeros((N, H, W, growth_rate))
        for n in range(N):
            for h in range(H):
                for w in range(W):
                    for g in range(growth_rate):
                        patch = padded[n, h:h+kh, w:w+kw, :]
                        conv_out[n, h, w, g] = np.sum(patch * kernel[:, :, :, g])

        # Concatenate along the channel axis
        features = np.concatenate([features, conv_out], axis=-1)

    return features
