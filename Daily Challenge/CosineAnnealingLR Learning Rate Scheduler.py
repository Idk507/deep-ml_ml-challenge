"""
Write a Python class CosineAnnealingLRScheduler to implement a learning rate scheduler based on the Cosine Annealing LR strategy. Your class should have an init method to initialize with an initial_lr (float), T_max (int, the maximum number of iterations/epochs), and min_lr (float, the minimum learning rate) parameters. It should also have a get_lr(self, epoch) method that returns the current learning rate for a given epoch (int). The learning rate should follow a cosine annealing schedule. The returned learning rate should be rounded to 4 decimal places. Only use standard Python and the math module for trigonometric functions.

Example:
Input:
import math
scheduler = CosineAnnealingLRScheduler(initial_lr=0.1, T_max=10, min_lr=0.001)
print(f"{scheduler.get_lr(epoch=0):.4f}")
print(f"{scheduler.get_lr(epoch=2):.4f}")
print(f"{scheduler.get_lr(epoch=5):.4f}")
print(f"{scheduler.get_lr(epoch=7):.4f}")
print(f"{scheduler.get_lr(epoch=10):.4f}")
Output:
0.1000
0.0905
0.0505
0.0214
0.0010
Reasoning:
The learning rate starts at initial_lr (0.1), follows a cosine curve, reaches min_lr (0.001) at T_max (epoch 10), and then cycles back up. Each value is rounded to 4 decimal places.


Learning Rate Schedulers: CosineAnnealingLR
1. Definition
A learning rate scheduler is a technique used in machine learning to adjust the learning rate during the training of a model. The learning rate dictates the step size taken in the direction of the negative gradient of the loss function.

CosineAnnealingLR (Cosine Annealing Learning Rate) is a scheduler that aims to decrease the learning rate from a maximum value to a minimum value following the shape of a cosine curve. This approach helps in achieving faster convergence while also allowing the model to explore flatter regions of the loss landscape towards the end of training. It is particularly effective for deep neural networks.

2. Why Use Learning Rate Schedulers?
Faster Convergence: A higher initial learning rate allows for quicker movement through the loss landscape.
Improved Performance: A smaller learning rate towards the end of training enables finer adjustments, helping the model converge to a better local minimum and preventing oscillations.
Avoiding Local Minima: The cyclical nature (or a part of it, as often seen in restarts) of cosine annealing can help the optimizer escape shallow local minima.
Stability: Gradual reduction in learning rate promotes training stability.
3. CosineAnnealingLR Mechanism
The learning rate is scheduled according to a cosine function. Over a cycle of 
T
max
T 
max
​
  epochs, the learning rate decreases from an initial learning rate (often considered the maximum 
L
R
max
LR 
max
​
 ) to a minimum learning rate (
L
R
min
LR 
min
​
 ).

The formula for the learning rate at a given epoch e is:

L
R
e
=
L
R
min
+
0.5
×
(
L
R
initial
−
L
R
min
)
×
(
1
+
cos
⁡
(
e
T
max
×
π
)
)
LR 
e
​
 =LR 
min
​
 +0.5×(LR 
initial
​
 −LR 
min
​
 )×(1+cos( 
T 
max
​
 
e
​
 ×π))

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
 : The initial (maximum) learning rate.
L
R
min
LR 
min
​
 : The minimum learning rate that the schedule will reach.
T
max
T 
max
​
 : The maximum number of epochs in the cosine annealing cycle. The learning rate will reach 
L
R
min
LR 
min
​
  at epoch 
T
max
T 
max
​
 .
e: The current epoch number (0-indexed), clamped between 0 and 
T
max
T 
max
​
 .
Ï: The mathematical constant pi (approximately 3.14159).
cos
⁡
(
⋅
)
cos(⋅): The cosine function.
Example: If 
L
R
initial
=
0.1
LR 
initial
​
 =0.1, 
T
max
=
10
T 
max
​
 =10, and 
L
R
min
=
0.001
LR 
min
​
 =0.001:

Epoch 0: 
L
R
0
=
0.001
+
0.5
×
(
0.1
−
0.001
)
×
(
1
+
cos
⁡
(
0
)
)
=
0.001
+
0.0495
×
2
=
0.1
LR 
0
​
 =0.001+0.5×(0.1−0.001)×(1+cos(0))=0.001+0.0495×2=0.1

Epoch 5 (mid-point): 
L
R
5
=
0.001
+
0.5
×
(
0.1
−
0.001
)
×
(
1
+
cos
⁡
(
π
/
2
)
)
=
0.001
+
0.0495
×
1
=
0.0505
LR 
5
​
 =0.001+0.5×(0.1−0.001)×(1+cos(π/2))=0.001+0.0495×1=0.0505

Epoch 10 (end of cycle): 
L
R
10
=
0.001
+
0.5
×
(
0.1
−
0.001
)
×
(
1
+
cos
⁡
(
π
)
)
=
0.001
+
0.0495
×
0
=
0.001
LR 
10
​
 =0.001+0.5×(0.1−0.001)×(1+cos(π))=0.001+0.0495×0=0.001

4. Applications of Learning Rate Schedulers
Learning rate schedulers, including CosineAnnealingLR, are widely used in training various machine learning models, especially deep neural networks, across diverse applications such as:

Image Classification: Training Convolutional Neural Networks (CNNs) for tasks like object recognition.
Natural Language Processing (NLP): Training Recurrent Neural Networks (RNNs) and Transformers for tasks like machine translation, text generation, and sentiment analysis.
Speech Recognition: Training models for converting spoken language to text.
Reinforcement Learning: Optimizing policies in reinforcement learning agents.
Any optimization problem where gradient descent or its variants are used.
"""

import math

class CosineAnnealingLRScheduler:
    def __init__(self, initial_lr, T_max, min_lr):
        # Initialize initial_lr, T_max, and min_lr
        self.initial_lr = initial_lr
        self.T_max = T_max 
        self.min_lr = min_lr 

    def get_lr(self, epoch):
        # Calculate and return the learning rate for the given epoch, rounded to 4 decimal places
        epoch = max(0,min(epoch,self.T_max))
        cosine_term = math.cos(math.pi * epoch/self.T_max)
        lr = self.min_lr + 0.5 * ( self.initial_lr - self.min_lr) * (1 + cosine_term)
        return round(lr,4)
