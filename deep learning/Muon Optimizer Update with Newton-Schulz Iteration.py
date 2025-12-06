import numpy as np

def newtonschulz5(G: np.ndarray, steps=5, eps=1e-7) -> np.ndarray:
    """
    Apply the Newton-Schulz (quintic) iteration for 5 steps to matrix G.
    Args:
        G: 2D NumPy array
        steps: Number of iteration steps (default 5)
        eps: Small constant for stability
    Returns:
        Matrix after Newton-Schulz iteration
    """
    # Normalize by Frobenius norm
    norm = np.linalg.norm(G, 'fro') + eps
    X = G / norm

    # If rows > cols, transpose before and after
    transposed = False
    if X.shape[0] > X.shape[1]:
        X = X.T
        transposed = True

    # Quintic coefficients
    a, b, c = 3.4445, -4.7750, 2.0315

    # Iteration
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X

    # Transpose back if needed
    if transposed:
        X = X.T

    return X


def muon_update(theta: np.ndarray, grad: np.ndarray, B_prev: np.ndarray, mu: float, lr: float) -> tuple:
    """
    Performs one Muon optimizer update (Algorithm 2).
    Returns the updated parameter, new B, and the preconditioned update.
    """
    # Step 1: Momentum update
    B_new = mu * B_prev + grad

    # Step 2: Preconditioning
    O = newtonschulz5(B_new)

    # Step 3: Parameter update
    theta_new = theta - lr * O

    return theta_new, B_new, O
