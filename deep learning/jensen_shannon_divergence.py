import numpy as np

def jensen_shannon_divergence(P: list[float], Q: list[float]) -> float:
    """
    Compute the Jensen-Shannon Divergence (JSD) between two probability distributions.

    The Jensen-Shannon Divergence is a symmetric and bounded measure of similarity
    between two probability distributions. It is defined as:
    
        JSD(P || Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M)
    
    where:
        M = 0.5 * (P + Q)
    
    Args:
        P (list[float]): First probability distribution (must sum to 1).
        Q (list[float]): Second probability distribution (must sum to 1).
    
    Returns:
        float: Jensen-Shannon Divergence value in natural logarithm units (nats),
               bounded between 0 and log(2).
    """
    # Convert inputs to numpy arrays for vectorized operations
    P = np.asarray(P, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)

    # Validate shapes
    if P.shape != Q.shape:
        raise ValueError("P and Q must have the same shape")

    # Validate probability distributions
    if np.any(P < 0) or np.any(Q < 0):
        raise ValueError("Probability distributions must be non-negative")

    # Normalize distributions to guard against floating-point drift
    P = P / P.sum()
    Q = Q / Q.sum()

    # Small epsilon for numerical stability (avoids log(0))
    eps = 1e-12
    P = np.clip(P, eps, 1.0)
    Q = np.clip(Q, eps, 1.0)

    # Compute the midpoint distribution
    M = 0.5 * (P + Q)

    # KL divergence components (vectorized, stable)
    kl_pm = np.sum(P * np.log(P / M))
    kl_qm = np.sum(Q * np.log(Q / M))

    # Jensen-Shannon Divergence
    jsd = 0.5 * (kl_pm + kl_qm)

    return float(jsd)
