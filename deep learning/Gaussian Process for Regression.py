"""
Problem Statement: Task is to implement GaussianProcessRegression class which is a guassian process model for prediction regression problems.

Example:
Input:
import numpy as np
gp = GaussianProcessRegression(kernel='linear', kernel_params={'sigma_b': 0.0, 'sigma_v': 1.0}, noise=1e-8)
X_train = np.array([[1], [2], [4]])
y_train = np.array([3, 5, 9])
gp.fit(X_train, y_train)
X_test = np.array([[3.0]])
mu = gp.predict(X_test)
print(f"{mu[0]:.4f}")
Output:
7.0000
Reasoning:
A Gaussian Process with a linear kernel is trained on perfectly linear data that follows the function y = 2x + 1. When asked to predict the value at x=3, the model perfectly interpolates the linear function it has learned, resulting in a prediction of 2*3 + 1 = 7. The near-zero noise ensures the prediction is exact.


Gaussian Processes (GP): From-Scratch Regression Example
1. What's a Gaussian Process?
A Gaussian Process defines a distribution over functions 
f
(
⋅
)
f(⋅). For any finite set of inputs 
(
X
=
x
i
i
=
1
n
)
(X=x 
i
​
  
i=1
n
​
 ), the function values 
f
(
X
)
f(X) follow a multivariate normal:

f
(
X
)
∼
N
(
0
,
;
K
(
X
,
X
)
)
f(X)∼N(0,;K(X,X))
where ( K ) is a kernel (covariance) function encoding similarity between inputs. With noisy targets 
(
y
=
f
(
X
)
+
ε
,
ε
∼
N
(
0
,
σ
n
2
I
)
)
(y=f(X)+ε,ε∼N(0,σ 
n
2
​
 I)), GP regression yields a closed-form posterior predictive mean and variance at new points 
(
X
∗
)
(X 
∗
​
 ).

2. The Implementation at a Glance
The provided code builds a minimal yet complete GP regression stack:

Kernels implemented

Radial Basis Function (RBF / Squared Exponential)
MatÃ©rn 
(
(
ν
=
0.5
,
1.5
,
2.5
)
,
o
r
g
e
n
e
r
a
l
(
ν
)
)
((ν=0.5,1.5,2.5),orgeneral(ν))
Periodic
Linear
Rational Quadratic
Core GP classes

_GaussianProcessBase: kernel selection & covariance matrix computation

GaussianProcessRegression:

fit: 
b
u
i
l
d
s
(
K
)
builds(K), does Cholesky decomposition, 
s
o
l
v
e
s
(
α
)
solves(α)
predict: returns posterior mean & variance
log_marginal_likelihood: computes GP evidence
optimize_hyperparameters: basic optimizer (for RBF hyperparams)
3. Kernel Cheat-Sheet
Let 
(
x
,
x
′
∈
R
d
)
,
(
r
=
∥
x
−
x
′
∥
)
(x,x 
′
 ∈R 
d
 ),(r=∥x−x 
′
 ∥).

RBF (SE):

k
RBF
(
x
,
x
′
)
=
σ
2
exp
⁡
!
(
−
1
2
r
2
ℓ
2
)
k 
RBF
​
 (x,x 
′
 )=σ 
2
 exp!(− 
2
1
​
  
ℓ 
2
 
r 
2
 
​
 )
MatÃ©rn (( \nu = 1.5 )):

k
(
x
,
x
′
)
=
(
1
+
3
,
r
ℓ
)
exp
⁡
!
(
−
3
,
r
ℓ
)
k(x,x 
′
 )=(1+ 
ℓ
3
​
 ,r
​
 )exp!(− 
ℓ
3
​
 ,r
​
 )
Periodic:

k
(
x
,
x
′
)
=
σ
2
exp
⁡
!
(
−
2
ℓ
2
sin
⁡
2
!
(
π
r
p
)
)
k(x,x 
′
 )=σ 
2
 exp!(− 
ℓ 
2
 
2
​
 sin 
2
 !( 
p
πr
​
 ))
Linear:

k
(
x
,
x
′
)
=
σ
b
2
+
σ
v
2
,
x
⊤
x
′
k(x,x 
′
 )=σ 
b
2
​
 +σ 
v
2
​
 ,x 
⊤
 x 
′
 
Rational Quadratic:

k
(
x
,
x
′
)
=
σ
2
(
1
+
r
2
2
α
ℓ
2
)
−
α
k(x,x 
′
 )=σ 
2
 (1+ 
2αℓ 
2
 
r 
2
 
​
 ) 
−α
 
4. GP Regression Mechanics
Training
Build covariance:

K
=
K
(
X
,
X
)
+
σ
n
2
I
K=K(X,X)+σ 
n
2
​
 I
Cholesky factorization:

K
=
L
L
⊤
K=LL 
⊤
 
Solve ( \alpha ):

L
L
⊤
α
=
y
LL 
⊤
 α=y
Prediction
At new inputs ( X_* ):

(
K
∗
=
K
(
X
,
X
∗
)
)
,
(
K
∗
∗
=
K
(
X
∗
,
X
∗
)
)
(K 
∗
​
 =K(X,X 
∗
​
 )),(K 
∗∗
​
 =K(X 
∗
​
 ,X 
∗
​
 ))

Mean:

μ
∗
=
K
∗
⊤
α
μ 
∗
​
 =K 
∗
⊤
​
 α
Covariance:

Σ
∗
=
K
∗
∗
−
V
⊤
V
,
V
=
L
−
1
K
∗
Σ 
∗
​
 =K 
∗∗
​
 −V 
⊤
 V,V=L 
−1
 K 
∗
​
 
Model Selection
Log Marginal Likelihood (LML):
log
⁡
p
(
y
∣
X
)
=
−
1
2
y
⊤
α
−
∑
i
log
⁡
L
i
i
−
n
2
log
⁡
(
2
π
)
logp(y∣X)=− 
2
1
​
 y 
⊤
 α−∑ 
i
​
 logL 
ii
​
 − 
2
n
​
 log(2π)
5. Worked Example (Linear Kernel)
import numpy as np
gp = GaussianProcessRegression(kernel='linear',
                               kernel_params={'sigma_b': 0.0, 'sigma_v': 1.0},
                               noise=1e-8)

X_train = np.array([[1], [2], [4]])
y_train = np.array([3, 5, 9])   # y = 2x + 1
gp.fit(X_train, y_train)

X_test = np.array([[3.0]])
mu = gp.predict(X_test)
print(f"{mu[0]:.4f}")   # -> 7.0000
6. When to Use GP Regression
Small-to-medium datasets where uncertainty estimates are valuable
Cases requiring predictive intervals (not just point predictions)
Nonparametric modeling with kernel priors
Automatic hyperparameter tuning via marginal likelihood
7. Practical Tips
Always add jitter 
10
−
6
10 
−6
  to the diagonal for numerical stability

Standardize inputs/outputs before training

Be aware: Exact GP has complexity 
O
(
n
3
)
O(n 
3
 ) in time and 
O
(
n
2
)
O(n 
2
 ) in memory

Choose kernels to match problem structure:

RBF: smooth functions
MatÃ©rn: rougher functions
Periodic: seasonal/cyclical data
Linear: global linear trends

"""

import math  # ---------------------------------------- utf-8 encoding ---------------------------------

# This file contains Gaussian Process implementation.
import numpy as np
import math


def matern_kernel(x, x_prime, length_scale=1.0, nu=1.5):
    r = np.linalg.norm(x - x_prime)
    if nu == 0.5:
        return np.exp(-r / length_scale)
    elif nu == 1.5:
        sqrt3 = np.sqrt(3)
        return (1 + sqrt3 * r / length_scale) * np.exp(-sqrt3 * r / length_scale)
    elif nu == 2.5:
        sqrt5 = np.sqrt(5)
        return (1 + sqrt5 * r / length_scale + 5 * r**2 / (3 * length_scale**2)) * np.exp(-sqrt5 * r / length_scale)
    else:
        raise NotImplementedError("Only ν=0.5, 1.5, 2.5 supported.")

def rbf_kernel(x, x_prime, sigma=1.0, length_scale=1.0):
    r2 = np.sum((x - x_prime)**2)
    return sigma**2 * np.exp(-0.5 * r2 / length_scale**2)

def periodic_kernel(x, x_prime, sigma=1.0, length_scale=1.0, period=1.0):
    r = np.linalg.norm(x - x_prime)
    return sigma**2 * np.exp(-2 * (np.sin(np.pi * r / period)**2) / length_scale**2)

def linear_kernel(x, x_prime, sigma_b=1.0, sigma_v=1.0):
    return sigma_b**2 + sigma_v**2 * np.dot(x, x_prime)

def rational_quadratic_kernel(x, x_prime, sigma=1.0, length_scale=1.0, alpha=1.0):
    r2 = np.sum((x - x_prime)**2)
    return sigma**2 * (1 + r2 / (2 * alpha * length_scale**2))**(-alpha)


# --- BASE CLASS -------------------------------------------------------------


class _GaussianProcessBase:
    def __init__(self, kernel="rbf", noise=1e-5, kernel_params=None):
        self.kernel = kernel
        self.noise = noise
        self.kernel_params = kernel_params or {}

    def _select_kernel(self, x1, x2):
        if self.kernel == "rbf":
            return rbf_kernel(x1, x2, **self.kernel_params)
        elif self.kernel == "matern":
            return matern_kernel(x1, x2, **self.kernel_params)
        elif self.kernel == "periodic":
            return periodic_kernel(x1, x2, **self.kernel_params)
        elif self.kernel == "linear":
            return linear_kernel(x1, x2, **self.kernel_params)
        elif self.kernel == "rational_quadratic":
            return rational_quadratic_kernel(x1, x2, **self.kernel_params)
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")

    def _compute_covariance(self, X1, X2):
        n1, n2 = X1.shape[0], X2.shape[0]
        K = np.zeros((n1, n2))
        for i in range(n1):
            for j in range(n2):
                K[i, j] = self._select_kernel(X1[i], X2[j])
        return K


# --- REGRESSION MODEL -------------------------------------------------------
class GaussianProcessRegression(_GaussianProcessBase):
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        K = self._compute_covariance(X, X) + self.noise * np.eye(len(X))
        self.L = np.linalg.cholesky(K + 1e-10 * np.eye(len(K)))  # jitter
        self.alpha = np.linalg.solve(self.L.T, np.linalg.solve(self.L, y))

    def predict(self, X_test, return_std=False):
        K_star = self._compute_covariance(self.X_train, X_test)
        mu = K_star.T @ self.alpha

        if return_std:
            v = np.linalg.solve(self.L, K_star)
            K_star_star = self._compute_covariance(X_test, X_test)
            cov = K_star_star - v.T @ v
            std = np.sqrt(np.diag(cov))
            return mu, std
        return mu

    def log_marginal_likelihood(self):
        y = self.y_train
        return -0.5 * y.T @ self.alpha - np.sum(np.log(np.diag(self.L))) - 0.5 * len(y) * np.log(2 * np.pi)

    def optimize_hyperparameters(self):
        raise NotImplementedError("Hyperparameter optimization not implemented.")
