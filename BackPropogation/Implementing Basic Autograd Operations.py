"""
Special thanks to Andrej Karpathy for making a video about this, if you haven't already check out his videos on YouTube https://youtu.be/VMj-3S1tku0?si=gjlnFP4o3JRN9dTg. Write a Python class similar to the provided 'Value' class that implements the basic autograd operations: addition, multiplication, and ReLU activation. The class should handle scalar values and should correctly compute gradients for these operations through automatic differentiation.

Example:
Input:
a = Value(2)
        b = Value(-3)
        c = Value(10)
        d = a + b * c
        e = d.relu()
        e.backward()
        print(a, b, c, d, e)
Output:
Value(data=2, grad=0) Value(data=-3, grad=0) Value(data=10, grad=0)
Reasoning:
The output reflects the forward computation and gradients after backpropagation. The ReLU on 'd' zeros out its output and gradient due to the negative data value.


Understanding Mathematical Concepts in Autograd Operations
First, watch the video in the Solution section.

This task focuses on implementing basic automatic differentiation mechanisms for neural networks. The operations of addition, multiplication, and ReLU are fundamental to neural network computations and their training through backpropagation.

Mathematical Foundations
Addition (__add__)

Forward Pass: For two scalar values ( a ) and ( b ), their sum ( s ) is:
s
=
a
+
b
s=a+b
Backward Pass: The derivative of ( s ) with respect to both ( a ) and ( b ) is 1. During backpropagation, the gradient of the output is passed directly to both inputs.
Multiplication (__mul__)

Forward Pass: For two scalar values ( a ) and ( b ), their product ( p ) is:
p
=
a
×
b
p=a×b
Backward Pass: The gradient of ( p ) with respect to ( a ) is ( b ), and with respect to ( b ) is ( a ). During backpropagation, each input's gradient is the product of the other input and the output's gradient.
ReLU Activation (relu)

Forward Pass: The ReLU function is defined as:
R
(
x
)
=
max
⁡
(
0
,
x
)
R(x)=max(0,x)
This function outputs ( x ) if ( x ) is positive, and 0 otherwise.
Backward Pass: The derivative of the ReLU function is 1 for ( x > 0 ) and 0 for ( x \leq 0 ). The gradient is propagated through the function only if the input is positive; otherwise, it stops.
Conceptual Application in Neural Networks
Addition and Multiplication: These operations are ubiquitous in neural networks, forming the basis for computing weighted sums of inputs in the neurons.
ReLU Activation: Commonly used as an activation function in neural networks due to its simplicity and effectiveness in introducing non-linearity, making learning complex patterns possible.
Understanding these operations and their implications on gradient flow is crucial for designing and training effective neural network models. By implementing these from scratch, you gain deeper insights into the workings of more sophisticated deep learning libraries

"""


class Value:
    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')
        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward
        return out

    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1
        for v in reversed(topo):
            v._backward()
    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"
