"""
Write a Python class ExponentialLRScheduler to implement a learning rate scheduler based on the ExponentialLR strategy. Your class should have an init method to initialize with an initial_lr (float) and gamma (float) parameter. It should also have a get_lr(self, epoch) method that returns the current learning rate for a given epoch (int). The learning rate should be decreased by gamma every epoch. The returned learning rate should be rounded to 4 decimal places. Only use standard Python.

Example:
Input:
scheduler = ExponentialLRScheduler(initial_lr=0.1, gamma=0.9)
print(f"{scheduler.get_lr(epoch=0):.4f}")
print(f"{scheduler.get_lr(epoch=1):.4f}")
print(f"{scheduler.get_lr(epoch=2):.4f}")
print(f"{scheduler.get_lr(epoch=3):.4f}")
Output:
0.1000
0.0900
0.0810
0.0729
Reasoning:
The initial learning rate is 0.1. At epoch 1, it decays by 0.9 to 0.09. At epoch 2, it decays again to 0.081, and so on, decaying by gamma every single epoch. All results are rounded to 4 decimal places.

Learning Rate Schedulers: ExponentialLR
1. Definition
A learning rate scheduler is a component used in machine learning, especially in neural network training, to adjust the learning rate during the training process. The learning rate is a hyperparameter that determines the step size at each iteration while moving towards a minimum of a loss function.

ExponentialLR (Exponential Learning Rate) is a common type of learning rate scheduler that decays the learning rate by a fixed multiplicative factor gamma at every epoch. This results in an exponential decrease of the learning rate over time. It's often used when a rapid and continuous reduction of the learning rate is desired.

2. Why Use Learning Rate Schedulers?
Faster Convergence: A higher initial learning rate can help quickly move through the loss landscape.
Improved Performance: A smaller learning rate towards the end of training allows for finer adjustments and helps in converging to a better local minimum, avoiding oscillations around the minimum.
Stability: Reducing the learning rate prevents large updates that could lead to divergence or instability.
3. ExponentialLR Mechanism
The learning rate is reduced by a factor gamma every epoch.

The formula for the learning rate at a given epoch e is:

L
R
e
=
L
R
initial
×
γ
e
LR 
e
​
 =LR 
initial
​
 ×γ 
e
 

Where:

L
R
e
LR 
e
​
 : The learning rate at epoch e.
L
R
initial
LR 
initial
​
 : The initial learning rate.
gamma: The multiplicative factor by which the learning rate is reduced per epoch (usually between 0 and 1, e.g., 0.9, 0.99).
e: The current epoch number (0-indexed).
Example: If initial learning rate = 0.1, and gamma = 0.9:

Epoch 0: 
L
R
0
=
0.1
×
0.9
0
=
0.1
×
1
=
0.1
LR 
0
​
 =0.1×0.9 
0
 =0.1×1=0.1
Epoch 1: 
L
R
1
=
0.1
×
0.9
1
=
0.1
×
0.9
=
0.09
LR 
1
​
 =0.1×0.9 
1
 =0.1×0.9=0.09
Epoch 2: 
L
R
2
=
0.1
×
0.9
2
=
0.1
×
0.81
=
0.081
LR 
2
​
 =0.1×0.9 
2
 =0.1×0.81=0.081
Epoch 3: 
L
R
3
=
0.1
×
0.9
3
=
0.1
×
0.729
=
0.0729
LR 
3
​
 =0.1×0.9 
3
 =0.1×0.729=0.0729
4. Applications of Learning Rate Schedulers
Learning rate schedulers, including ExponentialLR, are widely used in training various machine learning models, especially deep neural networks, across diverse applications such as:

Image Classification: Training Convolutional Neural Networks (CNNs) for tasks like object recognition.
Natural Language Processing (NLP): Training Recurrent Neural Networks (RNNs) and Transformers for tasks like machine translation, text generation, and sentiment analysis.
Speech Recognition: Training models for converting spoken language to text.
Reinforcement Learning: Optimizing policies in reinforcement learning agents.
Any optimization problem where gradient descent or its variants are used.

"""

class ExponentialLRScheduler:
    def __init__(self, initial_lr, gamma):
        # Store the initial learning rate and decay factor
        self.initial_lr = initial_lr
        self.gamma = gamma

    def get_lr(self, epoch):
        # Compute the exponentially decayed learning rate
        lr = self.initial_lr * (self.gamma ** epoch)
        # Round to 4 decimal places and return
        return round(lr, 4)

