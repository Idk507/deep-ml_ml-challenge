
import numpy as np
from typing import Callable, Dict, Optional, Tuple

# -----------------------------
# Kernels
# -----------------------------

def _ensure_2d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        return x.reshape(-1, 1)
    return x

def rbf_kernel(x: np.ndarray, x_prime: np.ndarray, sigma: float = 1.0, length_scale: float = 1.0) -> float:
    x = _ensure_2d(x)
    x_prime = _ensure_2d(x_prime)
    r2 = np.sum((x - x_prime) ** 2)
    return (sigma ** 2) * np.exp(-0.5 * r2 / (length_scale ** 2))

def matern_kernel(x: np.ndarray, x_prime: np.ndarray, length_scale: float = 1.0, nu: float = 1.5, sigma: float = 1.0) -> float:
    # Common closed forms for nu in {0.5, 1.5, 2.5}. Fallback to nu=1.5 if other values are requested (approx).
    x = _ensure_2d(x)
    x_prime = _ensure_2d(x_prime)
    r = np.sqrt(np.sum((x - x_prime) ** 2))
    if r == 0.0:
        return sigma ** 2
    z = np.sqrt(2 * nu) * r / length_scale

    if abs(nu - 0.5) < 1e-12:
        return (sigma ** 2) * np.exp(-z)
    elif abs(nu - 1.5) < 1e-12 or nu is None:
        return (sigma ** 2) * (1.0 + z) * np.exp(-z)
    elif abs(nu - 2.5) < 1e-12:
        return (sigma ** 2) * (1.0 + z + (z ** 2) / 3.0) * np.exp(-z)
    else:
        # Simple approximation using nu=1.5 form for general nu to keep dependency-free
        return (sigma ** 2) * (1.0 + z) * np.exp(-z)

def periodic_kernel(
    x: np.ndarray,
    x_prime: np.ndarray,
    sigma: float = 1.0,
    length_scale: float = 1.0,
    period: float = 1.0,
) -> float:
    x = _ensure_2d(x)
    x_prime = _ensure_2d(x_prime)
    r = np.sqrt(np.sum((x - x_prime) ** 2))
    s = np.sin(np.pi * r / period)
    return (sigma ** 2) * np.exp(-2 * (s ** 2) / (length_scale ** 2))

def linear_kernel(x, x_prime, sigma_b=1.0, sigma_v=1.0):
    x = np.asarray(x, dtype=float).reshape(-1)
    x_prime = np.asarray(x_prime, dtype=float).reshape(-1)
    dot = float(np.dot(x, x_prime))
    return (sigma_b**2) + (sigma_v**2) * dot


def rational_quadratic_kernel(
    x: np.ndarray,
    x_prime: np.ndarray,
    sigma: float = 1.0,
    length_scale: float = 1.0,
    alpha: float = 1.0,
) -> float:
    x = _ensure_2d(x)
    x_prime = _ensure_2d(x_prime)
    r2 = np.sum((x - x_prime) ** 2)
    return (sigma ** 2) * (1.0 + r2 / (2.0 * alpha * (length_scale ** 2))) ** (-alpha)


# -----------------------------
# Base GP
# -----------------------------

class _GaussianProcessBase:
    def __init__(self, kernel: str = "rbf", noise: float = 1e-5, kernel_params: Optional[Dict] = None, jitter: float = 1e-8):
        self.kernel_name = kernel.lower()
        self.noise = float(noise)
        self.kernel_params = kernel_params or {}
        self.jitter = float(jitter)

        # Validate kernel name
        valid = {"rbf", "matern", "periodic", "linear", "rational_quadratic"}
        if self.kernel_name not in valid:
            raise ValueError(f"Unsupported kernel '{kernel}'. Choose from {sorted(valid)}.")

    def _select_kernel(self) -> Callable[[np.ndarray, np.ndarray], float]:
        if self.kernel_name == "rbf":
            return lambda a, b: rbf_kernel(a, b, **self.kernel_params)
        if self.kernel_name == "matern":
            return lambda a, b: matern_kernel(a, b, **self.kernel_params)
        if self.kernel_name == "periodic":
            return lambda a, b: periodic_kernel(a, b, **self.kernel_params)
        if self.kernel_name == "linear":
            return lambda a, b: linear_kernel(a, b, **self.kernel_params)
        if self.kernel_name == "rational_quadratic":
            return lambda a, b: rational_quadratic_kernel(a, b, **self.kernel_params)
        raise RuntimeError("Kernel not configured.")

    def _compute_covariance(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        X1 = _ensure_2d(X1)
        X2 = _ensure_2d(X2)
        n1, n2 = X1.shape[0], X2.shape[0]
        K = np.empty((n1, n2), dtype=float)
        kfun = self._select_kernel()
        # Loop is simple and reliable for heterogeneous kernels; for RBF we could vectorize
        for i in range(n1):
            xi = X1[i : i + 1, :]
            for j in range(n2):
                xj = X2[j : j + 1, :]
                K[i, j] = kfun(xi, xj)
        return K


# -----------------------------
# GP Regression
# -----------------------------

class GaussianProcessRegression(_GaussianProcessBase):
    def fit(self, X: np.ndarray, y: np.ndarray):
        X = _ensure_2d(X)
        y = np.asarray(y, dtype=float).reshape(-1)
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of rows.")
        self.X_train = X
        self.y_train = y
        K = self._compute_covariance(X, X)
        K[np.diag_indices_from(K)] += self.noise + self.jitter
        self.L = np.linalg.cholesky(K)  # K = L L^T

        # Solve for alpha: alpha = K^{-1} y via two triangular solves
        # First solve L v = y
        v = np.linalg.solve(self.L, y)
        # Then solve L^T alpha = v
        self.alpha = np.linalg.solve(self.L.T, v)
        self._K = K  # For diagnostics
        return self

    def predict(self, X_test: np.ndarray, return_std: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if not hasattr(self, "X_train"):
            raise RuntimeError("Call fit before predict.")
        Xs = _ensure_2d(X_test)

        K_s = self._compute_covariance(self.X_train, Xs)       # shape (n_train, n_test)
        mu = K_s.T @ self.alpha                                 # shape (n_test,)

        if not return_std:
            return mu

        # Compute predictive variance diag: k_ss - v^T v, where v = L^{-1} K_s
        v = np.linalg.solve(self.L, K_s)                        # shape (n_train, n_test)
        K_ss = self._compute_covariance(Xs, Xs)
        var = np.clip(np.diag(K_ss) - np.sum(v * v, axis=0), a_min=0.0, a_max=None)
        std = np.sqrt(var)
        return mu, std

    def log_marginal_likelihood(self) -> float:
        if not hasattr(self, "L"):
            raise RuntimeError("Model not fitted.")
        y = self.y_train
        term1 = -0.5 * float(y @ self.alpha)
        term2 = -np.sum(np.log(np.diag(self.L)))
        n = y.shape[0]
        term3 = -0.5 * n * np.log(2.0 * np.pi)
        return term1 + term2 + term3

    def optimize_hyperparameters(self, kernel: Optional[str] = None, grid_sizes: Tuple[int, int] = (20, 20), length_scale_bounds=(1e-2, 1e2), sigma_bounds=(1e-2, 1e2)) -> Dict:
        """
        Very simple grid search for RBF hyperparameters (sigma, length_scale).
        Updates self.kernel_params with the best found values. Returns the best params.
        """
        kname = (kernel or self.kernel_name).lower()
        if kname != "rbf":
            raise NotImplementedError("optimize_hyperparameters currently supports only the RBF kernel.")

        sigmas = np.logspace(np.log10(sigma_bounds[0]), np.log10(sigma_bounds[1]), grid_sizes[0])
        ells = np.logspace(np.log10(length_scale_bounds[0]), np.log10(length_scale_bounds[1]), grid_sizes[1])

        best_lml = -np.inf
        best = None
        X = self.X_train
        y = self.y_train

        for s in sigmas:
            for ell in ells:
                self.kernel_params.update({"sigma": float(s), "length_scale": float(ell)})
                K = self._compute_covariance(X, X)
                K[np.diag_indices_from(K)] += self.noise + self.jitter
                try:
                    L = np.linalg.cholesky(K)
                except np.linalg.LinAlgError:
                    continue
                v = np.linalg.solve(L, y)
                alpha = np.linalg.solve(L.T, v)
                lml = (-0.5 * float(y @ alpha)) - np.sum(np.log(np.diag(L))) - 0.5 * X.shape[0] * np.log(2.0 * np.pi)
                if lml > best_lml:
                    best_lml = lml
                    best = {"sigma": float(s), "length_scale": float(ell)}

        if best is None:
            raise RuntimeError("Failed to find stable hyperparameters on the provided grid.")
        self.kernel_params.update(best)
        # Refit with best params
        self.fit(X, y)
        return best
