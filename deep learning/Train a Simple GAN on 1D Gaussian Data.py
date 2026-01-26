"""
In this task, you will train a Generative Adversarial Network (GAN) to learn a one-dimensional Gaussian distribution. The GAN consists of a generator that produces samples from latent noise and a discriminator that estimates the probability that a given sample is real. Both networks should have one hidden layer with ReLU activation in the hidden layer. The generatorâs output layer is linear, while the discriminator's output layer uses a sigmoid activation.

You must train the GAN using the standard non-saturating GAN loss for the generator and binary cross-entropy loss for the discriminator. In the NumPy version, parameters should be updated using vanilla gradient descent. In the PyTorch version, parameters should be updated using stochastic gradient descent (SGD) with the specified learning rate. The training loop should alternate between updating the discriminator and the generator each iteration.

Your function must return the trained generator forward function gen_forward(z), which produces generated samples given latent noise.

Example:
Input:
gen_forward = train_gan(4.0, 1.25, epochs=1000, seed=42)
z = np.random.normal(0, 1, (500, 1))
x_gen, _, _ = gen_forward(z)
(round(np.mean(x_gen), 4), round(np.std(x_gen), 4))
Output:
(0.0004, 0.0002)
Reasoning:
The test cases call gen_forward after training, sample 500 points, and then compute the mean and std.

Learn About topic
Understanding GANs for 1D Gaussian Data
A Generative Adversarial Network (GAN) consists of two neural networks - a Generator 
G
θ
G 
θ
​
  and a Discriminator 
D
ϕ
D 
ϕ
​
  - trained in a minimax game.

1. The Roles
Generator 
G
θ
(
z
)
G 
θ
​
 (z): Takes a latent noise vector 
z
∼
N
(
0
,
I
)
z∼N(0,I) and outputs a sample intended to resemble the real data.
Discriminator 
D
ϕ
(
x
)
D 
ϕ
​
 (x): Outputs a probability 
p
∈
(
0
,
1
)
p∈(0,1) that the input 
x
x came from the real data distribution rather than the generator.
2. The Objective
The classical GAN objective is:

min
⁡
θ
  
max
⁡
ϕ
  
E
x
∼
p
data
[
log
⁡
D
ϕ
(
x
)
]
+
E
z
∼
p
(
z
)
[
log
⁡
(
1
−
D
ϕ
(
G
θ
(
z
)
)
)
]
θ
min
​
  
ϕ
max
​
 E 
x∼p 
data
​
 
​
 [logD 
ϕ
​
 (x)]+E 
z∼p(z)
​
 [log(1−D 
ϕ
​
 (G 
θ
​
 (z)))]
Here:

p
data
p 
data
​
  is the real data distribution.
p
(
z
)
p(z) is the prior distribution for the latent noise (often standard normal).
3. Practical Losses
In implementation, we minimize:

Discriminator loss:
L
D
=
−
(
1
m
∑
i
=
1
m
log
⁡
D
(
x
real
(
i
)
)
+
log
⁡
(
1
−
D
(
x
fake
(
i
)
)
)
)
L 
D
​
 =−( 
m
1
​
  
i=1
∑
m
​
 logD(x 
real
(i)
​
 )+log(1−D(x 
fake
(i)
​
 )))
Generator loss (non-saturating form):
L
G
=
−
1
m
∑
i
=
1
m
log
⁡
D
(
G
(
z
(
i
)
)
)
L 
G
​
 =− 
m
1
​
  
i=1
∑
m
​
 logD(G(z 
(i)
 ))
4. Forward/Backward Flow
Discriminator step: Real samples 
x
real
x 
real
​
  and fake samples 
x
fake
=
G
(
z
)
x 
fake
​
 =G(z) are passed through 
D
D, and 
L
D
L 
D
​
  is minimized w.r.t. 
ϕ
ϕ.
Generator step: Fresh 
z
z is sampled, 
x
fake
=
G
(
z
)
x 
fake
​
 =G(z) is passed through 
D
D, and 
L
G
L 
G
​
  is minimized w.r.t. 
θ
θ while keeping 
ϕ
ϕ fixed.
5. Architecture for This Task
Generator: Fully connected layer (
R
latent_dim
→
R
hidden_dim
R 
latent_dim
 →R 
hidden_dim
 ) -> ReLU -> Fully connected layer (
R
hidden_dim
→
R
1
R 
hidden_dim
 →R 
1
 ).
Discriminator: Fully connected layer (
R
1
→
R
hidden_dim
R 
1
 →R 
hidden_dim
 ) â ReLU â Fully connected layer (
R
hidden_dim
→
R
1
R 
hidden_dim
 →R 
1
 ) â Sigmoid.
6. Numerical Tips
Initialize weights with a small Gaussian (
N
(
0
,
0.01
)
N(0,0.01)).
Add 
10
−
8
10 
−8
  to logs for numerical stability.
Use a consistent batch size 
m
m for both real and fake samples.
Always sample fresh noise for the generator on each update.
Your Task: Implement the training loop to learn the parameters 
θ
θ and 
ϕ
ϕ, and return the trained gen_forward(z) function. The evaluation (mean/std of generated samples) will be handled in the test cases.

"""
import numpy as np

def train_gan(mean_real: float, std_real: float, latent_dim: int = 1, hidden_dim: int = 16, learning_rate: float = 0.001, epochs: int = 5000, batch_size: int = 128, seed: int = 42):
    """
    Train a simple GAN to learn a 1D Gaussian distribution.

    Args:
        mean_real: Mean of the target Gaussian
        std_real: Std of the target Gaussian
        latent_dim: Dimension of the noise input to the generator
        hidden_dim: Hidden layer size for both networks
        learning_rate: Learning rate for gradient descent
        epochs: Number of training epochs
        batch_size: Training batch size
        seed: Random seed for reproducibility

    Returns:
        gen_forward: A function that takes z and returns generated samples
    """
    np.random.seed(seed)
    eps = 1e-8  # For numerical stability
    
    # Initialize Generator parameters: z -> hidden -> output
    # G_W1: (latent_dim, hidden_dim), G_b1: (hidden_dim,)
    # G_W2: (hidden_dim, 1), G_b2: (1,)
    G_W1 = np.random.normal(0, 0.01, (latent_dim, hidden_dim))
    G_b1 = np.zeros(hidden_dim)
    G_W2 = np.random.normal(0, 0.01, (hidden_dim, 1))
    G_b2 = np.zeros(1)
    
    # Initialize Discriminator parameters: x -> hidden -> output
    # D_W1: (1, hidden_dim), D_b1: (hidden_dim,)
    # D_W2: (hidden_dim, 1), D_b2: (1,)
    D_W1 = np.random.normal(0, 0.01, (1, hidden_dim))
    D_b1 = np.zeros(hidden_dim)
    D_W2 = np.random.normal(0, 0.01, (hidden_dim, 1))
    D_b2 = np.zeros(1)
    
    # Training loop
    for epoch in range(epochs):
        # Sample real data from target Gaussian
        x_real = np.random.normal(mean_real, std_real, (batch_size, 1))
        
        # Sample latent noise
        z = np.random.normal(0, 1, (batch_size, latent_dim))
        
        # ============= Generator Forward =============
        # Generate fake samples
        G_h1 = np.dot(z, G_W1) + G_b1  # (batch_size, hidden_dim)
        G_a1 = np.maximum(0, G_h1)  # ReLU
        x_fake = np.dot(G_a1, G_W2) + G_b2  # (batch_size, 1)
        
        # ============= Discriminator Forward (for training D) =============
        # Pass real samples through discriminator
        D_h1_real = np.dot(x_real, D_W1) + D_b1  # (batch_size, hidden_dim)
        D_a1_real = np.maximum(0, D_h1_real)  # ReLU
        D_logits_real = np.dot(D_a1_real, D_W2) + D_b2  # (batch_size, 1)
        D_out_real = 1 / (1 + np.exp(-D_logits_real))  # Sigmoid
        
        # Pass fake samples through discriminator
        D_h1_fake = np.dot(x_fake, D_W1) + D_b1  # (batch_size, hidden_dim)
        D_a1_fake = np.maximum(0, D_h1_fake)  # ReLU
        D_logits_fake = np.dot(D_a1_fake, D_W2) + D_b2  # (batch_size, 1)
        D_out_fake = 1 / (1 + np.exp(-D_logits_fake))  # Sigmoid
        
        # ============= Discriminator Loss & Backward =============
        # L_D = -[log(D(x_real)) + log(1 - D(x_fake))]
        # Gradient of BCE loss
        loss_D = -np.mean(np.log(D_out_real + eps) + np.log(1 - D_out_fake + eps))
        
        # Backprop for discriminator on real samples
        dL_dD_out_real = -1 / (D_out_real + eps) / batch_size
        dD_out_real = D_out_real * (1 - D_out_real)  # Sigmoid derivative
        dD_logits_real = dL_dD_out_real * dD_out_real
        
        dD_W2_real = np.dot(D_a1_real.T, dD_logits_real)
        dD_b2_real = np.sum(dD_logits_real, axis=0)
        dD_a1_real = np.dot(dD_logits_real, D_W2.T)
        dD_h1_real = dD_a1_real * (D_h1_real > 0)  # ReLU derivative
        dD_W1_real = np.dot(x_real.T, dD_h1_real)
        dD_b1_real = np.sum(dD_h1_real, axis=0)
        
        # Backprop for discriminator on fake samples
        dL_dD_out_fake = 1 / (1 - D_out_fake + eps) / batch_size
        dD_out_fake = D_out_fake * (1 - D_out_fake)
        dD_logits_fake = dL_dD_out_fake * dD_out_fake
        
        dD_W2_fake = np.dot(D_a1_fake.T, dD_logits_fake)
        dD_b2_fake = np.sum(dD_logits_fake, axis=0)
        dD_a1_fake = np.dot(dD_logits_fake, D_W2.T)
        dD_h1_fake = dD_a1_fake * (D_h1_fake > 0)
        dD_W1_fake = np.dot(x_fake.T, dD_h1_fake)
        dD_b1_fake = np.sum(dD_h1_fake, axis=0)
        
        # Update discriminator parameters
        D_W2 -= learning_rate * (dD_W2_real + dD_W2_fake)
        D_b2 -= learning_rate * (dD_b2_real + dD_b2_fake)
        D_W1 -= learning_rate * (dD_W1_real + dD_W1_fake)
        D_b1 -= learning_rate * (dD_b1_real + dD_b1_fake)
        
        # ============= Generator Update =============
        # Sample fresh noise for generator update
        z_gen = np.random.normal(0, 1, (batch_size, latent_dim))
        
        # Generator forward
        G_h1_gen = np.dot(z_gen, G_W1) + G_b1
        G_a1_gen = np.maximum(0, G_h1_gen)
        x_fake_gen = np.dot(G_a1_gen, G_W2) + G_b2
        
        # Discriminator forward on generated samples
        D_h1_gen = np.dot(x_fake_gen, D_W1) + D_b1
        D_a1_gen = np.maximum(0, D_h1_gen)
        D_logits_gen = np.dot(D_a1_gen, D_W2) + D_b2
        D_out_gen = 1 / (1 + np.exp(-D_logits_gen))
        
        # Generator loss (non-saturating): L_G = -log(D(G(z)))
        loss_G = -np.mean(np.log(D_out_gen + eps))
        
        # Backprop through discriminator (frozen) to generator
        dL_dD_out_gen = -1 / (D_out_gen + eps) / batch_size
        dD_out_gen = D_out_gen * (1 - D_out_gen)
        dD_logits_gen = dL_dD_out_gen * dD_out_gen
        
        dD_a1_gen = np.dot(dD_logits_gen, D_W2.T)
        dD_h1_gen = dD_a1_gen * (D_h1_gen > 0)
        dx_fake_gen = np.dot(dD_h1_gen, D_W1.T)
        
        # Backprop through generator
        dG_W2 = np.dot(G_a1_gen.T, dx_fake_gen)
        dG_b2 = np.sum(dx_fake_gen, axis=0)
        dG_a1_gen = np.dot(dx_fake_gen, G_W2.T)
        dG_h1_gen = dG_a1_gen * (G_h1_gen > 0)
        dG_W1 = np.dot(z_gen.T, dG_h1_gen)
        dG_b1 = np.sum(dG_h1_gen, axis=0)
        
        # Update generator parameters
        G_W1 -= learning_rate * dG_W1
        G_b1 -= learning_rate * dG_b1
        G_W2 -= learning_rate * dG_W2
        G_b2 -= learning_rate * dG_b2
    
    # Return generator forward function
    def gen_forward(z):
        G_h1 = np.dot(z, G_W1) + G_b1
        G_a1 = np.maximum(0, G_h1)
        x_gen = np.dot(G_a1, G_W2) + G_b2
        return x_gen, G_h1, G_a1
    
    return gen_forward
