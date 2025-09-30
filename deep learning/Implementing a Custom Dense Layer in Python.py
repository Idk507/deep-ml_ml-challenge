"""
Implementing a Custom Dense Layer in Python
You are provided with a base Layer class that defines the structure of a neural network layer. Your task is to implement a subclass called Dense, which represents a fully connected neural network layer. The Dense class should extend the Layer class and implement the following methods:

Initialization (__init__):

Define the layer with a specified number of neurons (n_units) and an optional input shape (input_shape).
Set up placeholders for the layer's weights (W), biases (w0), and optimizers.
Weight Initialization (initialize):

Initialize the weights W using a uniform distribution with a limit of 1 / sqrt(input_shape[0]), and bias w0 should be set to zero.
Initialize optimizers for W and w0.
Parameter Count (parameters):

Return the total number of trainable parameters in the layer, which includes the parameters in W and w0.
Forward Pass (forward_pass):

Compute the output of the layer by performing a dot product between the input X and the weight matrix W, and then adding the bias w0.
Backward Pass (backward_pass):

Calculate and return the gradient with respect to the input.
If the layer is trainable, update the weights and biases using the optimizer's update rule.
Output Shape (output_shape):

Return the shape of the output produced by the forward pass, which should be (self.n_units,).
Objective:
Extend the Layer class by implementing the Dense class to ensure it functions correctly within a neural network framework.

Example:
Input:
# Initialize a Dense layer with 3 neurons and input shape (2,)
dense_layer = Dense(n_units=3, input_shape=(2,))

# Define a mock optimizer with a simple update rule
class MockOptimizer:
    def update(self, weights, grad):
        return weights - 0.01 * grad

optimizer = MockOptimizer()

# Initialize the Dense layer with the mock optimizer
dense_layer.initialize(optimizer)

# Perform a forward pass with sample input data
X = np.array([[1, 2]])
output = dense_layer.forward_pass(X)
print("Forward pass output:", output)

# Perform a backward pass with sample gradient
accum_grad = np.array([[0.1, 0.2, 0.3]])
back_output = dense_layer.backward_pass(accum_grad)
print("Backward pass output:", back_output)
Output:
Forward pass output: [[-0.00655782  0.01429615  0.00905812]]
Backward pass output: [[ 0.00129588  0.00953634]]
Reasoning:
The code initializes a Dense layer with 3 neurons and input shape (2,). It then performs a forward pass with sample input data and a backward pass with sample gradients. The output demonstrates the forward and backward pass results.


Understanding the Dense Layer
The Dense layer, also known as a fully connected layer, is a fundamental building block in neural networks. It connects each input neuron to each output neuron, hence the term "fully connected."

1. Weight Initialization
In the initialize method, weights are typically initialized using a uniform distribution within a certain range. For a Dense layer, a common practice is to set this range as:

limit
=
1
input_shape
limit= 
input_shape
​
 
1
​
 
This initialization helps in maintaining a balance in the distribution of weights, preventing issues like vanishing or exploding gradients during training.

2. Forward Pass
During the forward pass, the input data ( X ) is multiplied by the weight matrix ( W ) and added to the bias ( w_0 ) to produce the output:

output
=
X
⋅
W
+
w
0
output=X⋅W+w 
0
​
 
3. Backward Pass
The backward pass computes the gradients of the loss function with respect to the input data, weights, and biases. If the layer is trainable, it updates the weights and biases using the optimizer's update rule:

W
=
W
−
η
⋅
grad
W
W=W−η⋅grad 
W
​
 
w
0
=
w
0
−
η
⋅
grad
w
0
w 
0
​
 =w 
0
​
 −η⋅grad 
w 
0
​
 
​
 
where ( \eta ) is the learning rate and ( \text{grad}W ) and ( \text{grad}{w_0} ) are the gradients of the weights and biases, respectively.

4. Output Shape
The shape of the output from a Dense layer is determined by the number of neurons in the layer. If a layer has n_units neurons, the output shape will be ( (n_units,) ).

Resources
CS231n: Fully Connected Layer

"""


class Dense(Layer):
    def __init__(self, n_units, input_shape=None):
        super().__init__()
        self.n_units = n_units
        self.input_shape = input_shape
        self.trainable = True
        self.W = None
        self.w0 = None
        self.optimizer_W = None
        self.optimizer_w0 = None
        self.layer_input = None

    def initialize(self, optimizer):
        limit = 1 / math.sqrt(self.input_shape[0])
        self.W = np.random.uniform(-limit, limit, (self.input_shape[0], self.n_units))
        self.w0 = np.zeros((1, self.n_units))
        self.optimizer_W = copy.deepcopy(optimizer)
        self.optimizer_w0 = copy.deepcopy(optimizer)

    def parameters(self):
        return self.W.size + self.w0.size

    def forward_pass(self, X, training=True):
        self.layer_input = X
        return np.dot(X, self.W) + self.w0

    def backward_pass(self, accum_grad):
        grad_W = np.dot(self.layer_input.T, accum_grad)
        grad_w0 = np.sum(accum_grad, axis=0, keepdims=True)
        back_output = np.dot(accum_grad, self.W.T)

        if self.trainable:
            self.W = self.optimizer_W.update(self.W, grad_W)
            self.w0 = self.optimizer_w0.update(self.w0, grad_w0)

        return back_output

    def output_shape(self):
        return (self.n_units,)
