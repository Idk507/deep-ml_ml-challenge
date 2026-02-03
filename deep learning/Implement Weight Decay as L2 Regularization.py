"""
Implement weight decay (L2 regularization) for neural network parameters. Given parameter arrays, their gradients, a learning rate, and a weight decay factor, apply the parameter update that includes both gradient descent and L2 regularization. The function should take a boolean list indicating which parameter groups should have weight decay applied (typically, weight decay is applied to weights but not to biases). Return the updated parameters after applying the appropriate update rule.

Example:
Input:
parameters=[[1.0, 2.0]], gradients=[[0.1, 0.2]], lr=0.1, weight_decay=0.01, apply_to_all=[True]
Output:
[[0.989, 1.978]]
Reasoning:
For the first parameter (1.0): 
1.0
−
0.1
×
0.1
−
0.1
×
0.01
×
1.0
=
1.0
−
0.01
−
0.001
=
0.989
1.0−0.1×0.1−0.1×0.01×1.0=1.0−0.01−0.001=0.989. For the second parameter (2.0): 
2.0
−
0.1
×
0.2
−
0.1
×
0.01
×
2.0
=
2.0
−
0.02
−
0.002
=
1.978
2.0−0.1×0.2−0.1×0.01×2.0=2.0−0.02−0.002=1.978.

Learn About topic
Understanding Weight Decay and L2 Regularization
Weight decay is a regularization technique used to prevent overfitting in neural networks by penalizing large weights. It's closely related to L2 regularization, though they're implemented slightly differently.

The Regularization Problem
Without regularization, neural networks can develop very large weights that lead to:

Overfitting: Model memorizes training data instead of learning generalizable patterns
Poor generalization: High training accuracy but low test accuracy
Numerical instability: Large weights can cause gradient explosions
L2 Regularization in Loss Function
L2 regularization adds a penalty term to the loss function:

L
t
o
t
a
l
=
L
o
r
i
g
i
n
a
l
+
λ
2
∑
i
w
i
2
L 
total
​
 =L 
original
​
 + 
2
λ
​
  
i
∑
​
 w 
i
2
​
 
Where:

L
o
r
i
g
i
n
a
l
L 
original
​
  is the original loss (e.g., cross-entropy)
λ
λ is the regularization strength (weight decay factor)
w
i
w 
i
​
  are the model weights
The 
1
2
2
1
​
  factor is for mathematical convenience in derivatives
Weight Decay Update Rule
When we take the gradient of the regularized loss and apply gradient descent, we get:

w
n
e
w
=
w
−
η
∂
L
t
o
t
a
l
∂
w
w 
new
​
 =w−η 
∂w
∂L 
total
​
 
​
 
Expanding the gradient:

∂
L
t
o
t
a
l
∂
w
=
∂
L
o
r
i
g
i
n
a
l
∂
w
+
λ
w
∂w
∂L 
total
​
 
​
 = 
∂w
∂L 
original
​
 
​
 +λw
Substituting back:

w
n
e
w
=
w
−
η
(
∂
L
o
r
i
g
i
n
a
l
∂
w
+
λ
w
)
w 
new
​
 =w−η( 
∂w
∂L 
original
​
 
​
 +λw)
This simplifies to:

w
n
e
w
=
w
−
η
⋅
g
−
η
⋅
λ
⋅
w
w 
new
​
 =w−η⋅g−η⋅λ⋅w
Where 
g
=
∂
L
o
r
i
g
i
n
a
l
∂
w
g= 
∂w
∂L 
original
​
 
​
  is the gradient.

This can be rewritten as:

w
n
e
w
=
(
1
−
η
λ
)
w
−
η
⋅
g
w 
new
​
 =(1−ηλ)w−η⋅g
The term 
(
1
−
η
λ
)
(1−ηλ) causes the weights to "decay" toward zero at each step, hence the name "weight decay."

Key Components
Parameters:

w
w: Current weight value
g
g: Gradient of the loss with respect to the weight
η
η: Learning rate
λ
λ: Weight decay factor (typically 0.0001 to 0.01)
Update Formula:

w_new = w - lr * grad - lr * weight_decay * w
Why Exclude Biases?
By convention, weight decay is not applied to bias terms. The reasons are:

Biases don't cause overfitting: Biases shift the activation function but don't amplify inputs like weights do
Common practice: Empirically, excluding biases from regularization works better
Theoretical justification: L2 regularization penalizes complexity; biases don't add model complexity
Example Walkthrough
Given:

Weight: 
w
=
1.0
w=1.0
Gradient: 
g
=
0.1
g=0.1
Learning rate: 
η
=
0.1
η=0.1
Weight decay: 
λ
=
0.01
λ=0.01
Step-by-step calculation:

Gradient descent term: 
η
⋅
g
=
0.1
×
0.1
=
0.01
η⋅g=0.1×0.1=0.01

Weight decay term: 
η
⋅
λ
⋅
w
=
0.1
×
0.01
×
1.0
=
0.001
η⋅λ⋅w=0.1×0.01×1.0=0.001

Updated weight: 
w
n
e
w
=
1.0
−
0.01
−
0.001
=
0.989
w 
new
​
 =1.0−0.01−0.001=0.989

L2 Regularization vs Weight Decay
While often used interchangeably, there's a subtle difference:

| Aspect | L2 Regularization | Weight Decay | |--------|------------------|-------------| | Implementation | Add penalty to loss | Directly modify weight update | | With momentum | Different behavior | Simpler, more effective | | With adaptive learning rates (Adam) | Not equivalent | Often preferred | | Mathematical equivalence | Only for standard SGD | Direct implementation |

For standard SGD, they're mathematically equivalent. For advanced optimizers (Adam, RMSprop), weight decay is often preferred as "decoupled weight decay."

Practical Tips
Typical values: Start with 
λ
=
0.0001
λ=0.0001 or 
0.01
0.01
Hyperparameter tuning: Weight decay is a crucial hyperparameter to tune
Layer-specific decay: Sometimes different layers use different decay rates
Monitor training: Too much decay â underfitting; too little â overfitting
Effect on Training
Without weight decay:

Weights can grow arbitrarily large
Model may overfit to training data
High variance in predictions
With weight decay:

Weights stay relatively small
Better generalization to unseen data
Smoother decision boundaries
More stable training
Implementation in Popular Frameworks
PyTorch: torch.optim.SGD(params, lr=0.1, weight_decay=0.01)
TensorFlow: tf.keras.optimizers.SGD(learning_rate=0.1, weight_decay=0.01)
Both frameworks automatically exclude biases by default (if configured properly)
Weight decay is one of the most effective and widely-used regularization techniques in deep learning, helping models generalize better while remaining simple to implement.


"""
def apply_weight_decay(parameters: list[list[float]], gradients: list[list[float]], 
                       lr: float, weight_decay: float, apply_to_all: list[bool]) -> list[list[float]]:
    """
    Apply weight decay (L2 regularization) to parameters.
    
    Args:
        parameters: List of parameter arrays
        gradients: List of gradient arrays
        lr: Learning rate
        weight_decay: Weight decay factor
        apply_to_all: Boolean list indicating which parameter groups get weight decay
    
    Returns:
        Updated parameters
    """
    updated_parameters = []
    
    # Iterate through each parameter group
    for i, (param_group, grad_group) in enumerate(zip(parameters, gradients)):
        updated_group = []
        
        # Check if weight decay should be applied to this parameter group
        apply_decay = apply_to_all[i]
        
        # Update each parameter in the group
        for param, grad in zip(param_group, grad_group):
            if apply_decay:
                # Apply weight decay: w_new = w - lr * grad - lr * weight_decay * w
                param_new = param - lr * grad - lr * weight_decay * param
            else:
                # Only apply gradient descent: w_new = w - lr * grad
                param_new = param - lr * grad
            
            updated_group.append(param_new)
        
        updated_parameters.append(updated_group)
    
    return updated_parameters
