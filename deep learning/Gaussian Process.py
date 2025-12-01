"""
Gaussian Process for Regression
Hard
Machine Learning

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

Learn About topic
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

# -*- coding: utf-8 -*-
import numpy as np
import math


# --- KERNELS ---------------------------------------------------------------

def _euclidean_distance(x: np.ndarray, x_prime: np.ndarray) -> float:
    # Assumes x, x_prime are 1D vectors
    diff = x - x_prime
    return float(np.sqrt(np.dot(diff, diff)))


def matern_kernel(x: np.ndarray, x_prime: np.ndarray, length_scale=1.0, nu=1.5, sigma=1.0):
    """
    Matérn kernel for nu in {0.5, 1.5, 2.5}. General nu not implemented without special functions.
    k(r) = sigma^2 * f_nu(r / length_scale)
    """
    r = _euclidean_distance(x, x_prime)
    if length_scale <= 0:
        raise ValueError("length_scale must be > 0")
    if sigma <= 0:
        raise ValueError("sigma must be > 0")

    if nu == 0.5:
        # Exponential kernel
        k = sigma ** 2 * math.exp(-r / length_scale)
    elif nu == 1.5:
        c = math.sqrt(3) * r / length_scale
        k = sigma ** 2 * (1.0 + c) * math.exp(-c)
    elif nu == 2.5:
        c = math.sqrt(5) * r / length_scale
        k = sigma ** 2 * (1.0 + c + (5.0 * r * r) / (3.0 * length_scale * length_scale)) * math.exp(-c)
    else:
        raise NotImplementedError("Matérn kernel implemented for nu in {0.5, 1.5, 2.5}.")
    return k


def rbf_kernel(x: np.ndarray, x_prime: np.ndarray, sigma=1.0, length_scale=1.0):
    """
    Squared Exponential (RBF) kernel:
    k(x, x') = sigma^2 * exp( -0.5 * ||x-x'||^2 / length_scale^2 )
    """
    if length_scale <= 0:
        raise ValueError("length_scale must be > 0")
    if sigma <= 0:
        raise ValueError("sigma must be > 0")
    r2 = float(np.dot(x - x_prime, x - x_prime))
    return sigma ** 2 * math.exp(-0.5 * r2 / (length_scale ** 2))


def periodic_kernel(x: np.ndarray, x_prime: np.ndarray, sigma=1.0, length_scale=1.0, period=1.0):
    """
    Periodic kernel:
    k(x, x') = sigma^2 * exp( -2 * sin^2( pi * ||x-x'|| / period ) / length_scale^2 )
    """
    if length_scale <= 0:
        raise ValueError("length_scale must be > 0")
    if sigma <= 0:
        raise ValueError("sigma must be > 0")
    if period <= 0:
        raise ValueError("period must be > 0")

    r = _euclidean_distance(x, x_prime)
    s = math.sin(math.pi * r / period)
    return sigma ** 2 * math.exp(-2.0 * (s ** 2) / (length_scale ** 2))


def linear_kernel(x: np.ndarray, x_prime: np.ndarray, sigma_b=1.0, sigma_v=1.0):
    """
    Linear kernel:
    k(x, x') = sigma_b^2 + sigma_v^2 * (x^T x')
    """
    if sigma_b < 0 or sigma_v < 0:
        raise ValueError("sigma_b and sigma_v must be >= 0")
    return (sigma_b ** 2) + (sigma_v ** 2) * float(np.dot(x, x_prime))


def rational_quadratic_kernel(x: np.ndarray, x_prime: np.ndarray, sigma=1.0, length_scale=1.0, alpha=1.0):
    """
    Rational Quadratic kernel:
    k(x, x') = sigma^2 * ( 1 + ||x-x'||^2 / (2*alpha*length_scale^2) )^{-alpha}
    """
    if length_scale <= 0:
        raise ValueError("length_scale must be > 0")
    if sigma <= 0:
        raise ValueError("sigma must be > 0")
    if alpha <= 0:
        raise ValueError("alpha must be > 0")

    r2 = float(np.dot(x - x_prime, x - x_prime))
    return sigma ** 2 * (1.0 + r2 / (2.0 * alpha * (length_scale ** 2))) ** (-alpha)


# --- BASE CLASS ------------------------------------------------------------

class _GaussianProcessBase:
    def __init__(self, kernel="rbf", noise=1e-5, kernel_params=None, jitter=1e-6):
        """
        kernel: one of {'rbf', 'matern', 'periodic', 'linear', 'rq'}
        noise: observation noise variance (added to diag of K)
        kernel_params: dict of kernel hyperparameters
        jitter: small diagonal addition for numerical stability
        """
        self.kernel = kernel.lower()
        self.noise = float(noise)
        self.kernel_params = kernel_params or {}
        self.jitter = float(jitter)
        if self.noise < 0:
            raise ValueError("noise must be >= 0")
        if self.jitter <= 0:
            raise ValueError("jitter must be > 0")

    def _select_kernel(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Selects and computes the kernel value for two single data points."""
        kp = self.kernel_params
        if self.kernel == "rbf":
            return rbf_kernel(x1, x2, sigma=kp.get("sigma", 1.0), length_scale=kp.get("length_scale", 1.0))
        elif self.kernel == "matern":
            return matern_kernel(
                x1, x2,
                length_scale=kp.get("length_scale", 1.0),
                nu=kp.get("nu", 1.5),
                sigma=kp.get("sigma", 1.0),
            )
        elif self.kernel == "periodic":
            return periodic_kernel(
                x1, x2,
                sigma=kp.get("sigma", 1.0),
                length_scale=kp.get("length_scale", 1.0),
                period=kp.get("period", 1.0),
            )
        elif self.kernel == "linear":
            return linear_kernel(x1, x2, sigma_b=kp.get("sigma_b", 1.0), sigma_v=kp.get("sigma_v", 1.0))
        elif self.kernel in ("rq", "rational_quadratic"):
            return rational_quadratic_kernel(
                x1, x2,
                sigma=kp.get("sigma", 1.0),
                length_scale=kp.get("length_scale", 1.0),
                alpha=kp.get("alpha", 1.0),
            )
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")

    def _ensure_2d(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return X

    def _compute_covariance(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """
        Computes the covariance matrix between two sets of points.
        X1: (n1, d), X2: (n2, d)
        Returns: (n1, n2) matrix K where K[i,j] = k(X1[i], X2[j])
        """
        X1 = self._ensure_2d(X1)
        X2 = self._ensure_2d(X2)
        n1, _ = X1.shape
        n2, _ = X2.shape
        K = np.empty((n1, n2), dtype=float)
        # Loop-based for clarity and correctness across kernels
        for i in range(n1):
            for j in range(n2):
                K[i, j] = self._select_kernel(X1[i], X2[j])
        return K


# --- REGRESSION MODEL ------------------------------------------------------

class GaussianProcessRegression(_GaussianProcessBase):
    def __init__(self, kernel="rbf", noise=1e-5, kernel_params=None, jitter=1e-6):
        super().__init__(kernel=kernel, noise=noise, kernel_params=kernel_params, jitter=jitter)
        self.X_train = None
        self.y_train = None
        self.L = None      # Cholesky factor of K
        self.alpha = None  # K^{-1} y via triangular solves

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the GP to training data.
        X: (n, d) or (n,)
        y: (n,)
        """
        X = self._ensure_2d(X)
        y = np.asarray(y, dtype=float).reshape(-1)
        n = X.shape[0]
        if y.shape[0] != n:
            raise ValueError("X and y must have the same number of samples")

        # Build covariance with noise + jitter on diagonal
        K = self._compute_covariance(X, X)
        diag_add = (self.noise + self.jitter)
        K[np.diag_indices(n)] += diag_add

        # Cholesky decomposition (ensure positive-definite via jitter)
        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            # Increase jitter and retry
            jitter = max(self.jitter, 1e-8)
            for _ in range(5):
                K[np.diag_indices(n)] += jitter
                try:
                    L = np.linalg.cholesky(K)
                    break
                except np.linalg.LinAlgError:
                    jitter *= 10.0
            else:
                raise np.linalg.LinAlgError("Cholesky decomposition failed. Consider adjusting jitter or kernel params.")

        # Solve for alpha: K alpha = y using triangular solves
        # L L^T alpha = y => solve L v = y, then L^T alpha = v
        v = np.linalg.solve(L, y)
        alpha = np.linalg.solve(L.T, v)

        # Store
        self.X_train = X
        self.y_train = y
        self.L = L
        self.alpha = alpha
        return self

    def predict(self, X_test: np.ndarray, return_std: bool = False):
        """
        Predict posterior mean (and optional std) at test points.
        X_test: (m, d) or (m,)
        """
        if self.X_train is None or self.alpha is None or self.L is None:
            raise RuntimeError("Model must be fitted before prediction.")

        X_test = self._ensure_2d(X_test)

        K_star = self._compute_covariance(self.X_train, X_test)       # (n, m)
        mu = K_star.T @ self.alpha                                    # (m,)

        if not return_std:
            return mu

        # Predictive variance: diag(K_** - V^T V), where V = L^{-1} K_star
        K_starstar = self._compute_covariance(X_test, X_test)         # (m, m)
        # Only diagonal is needed; compute via V
        V = np.linalg.solve(self.L, K_star)                           # (n, m)
        var = np.diag(K_starstar) - np.sum(V * V, axis=0)
        # Numerical safety: clip negative due to rounding
        var = np.clip(var, 0.0, None)
        std = np.sqrt(var)
        return mu, std

    def log_marginal_likelihood(self):
        """
        Compute the log marginal likelihood:
        LML = -0.5 * y^T alpha - sum(log(diag(L))) - n/2 * log(2*pi)
        """
        if self.X_train is None or self.alpha is None or self.L is None:
            raise RuntimeError("Model must be fitted before computing LML.")
        y = self.y_train
        L = self.L
        n = y.shape[0]
        term1 = -0.5 * float(y @ self.alpha)
        term2 = -np.sum(np.log(np.diag(L)))
        term3 = -0.5 * n * math.log(2.0 * math.pi)
        return term1 + term2 + term3

    def optimize_hyperparameters(self, length_scales=(0.1, 0.2, 0.5, 1.0, 2.0, 5.0), sigmas=(0.5, 1.0, 2.0, 5.0)):
        """
        Simple grid search optimizer for RBF kernel hyperparameters: length_scale and sigma.
        Keeps noise fixed. Re-fits the model using the best setting by LML.
        """
        if self.kernel != "rbf":
            raise NotImplementedError("optimize_hyperparameters implemented only for RBF kernel.")

        if self.X_train is None or self.y_train is None:
            raise RuntimeError("Fit the model with initial hyperparameters before optimization.")

        best_lml = -np.inf
        best_params = None
        X = self.X_train
        y = self.y_train

        # Preserve original params to restore or update
        original_params = dict(self.kernel_params)

        for ls in length_scales:
            for sg in sigmas:
                self.kernel_params["length_scale"] = float(ls)
                self.kernel_params["sigma"] = float(sg)
                # Rebuild K and factorization quickly (reuse fit steps without changing X, y)
                n = X.shape[0]
                K = self._compute_covariance(X, X)
                K[np.diag_indices(n)] += (self.noise + self.jitter)
                try:
                    L = np.linalg.cholesky(K)
                except np.linalg.LinAlgError:
                    # Skip unstable combinations
                    continue
                v = np.linalg.solve(L, y)
                alpha = np.linalg.solve(L.T, v)
                term1 = -0.5 * float(y @ alpha)
                term2 = -np.sum(np.log(np.diag(L)))
                term3 = -0.5 * n * math.log(2.0 * math.pi)
                lml = term1 + term2 + term3
                if lml > best_lml:
                    best_lml = lml
                    best_params = {"length_scale": float(ls), "sigma": float(sg)}
                    best_L = L
                    best_alpha = alpha

        if best_params is None:
            # Restore original params if nothing improved or stability failed
            self.kernel_params = original_params
            return original_params, best_lml

        # Update model with best params and cached factorization
        self.kernel_params.update(best_params)
        self.L = best_L
        self.alpha = best_alpha
        return best_params, best_lml


