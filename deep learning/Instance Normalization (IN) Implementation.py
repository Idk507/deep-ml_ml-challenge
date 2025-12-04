def instance_normalization(X, gamma, beta, eps=1e-5):
    """
    Instance Normalization for 4D tensors (B, C, H, W).

    Parameters:
    -----------
    X : np.ndarray
        Input tensor of shape (B, C, H, W).
    gamma : np.ndarray
        Scale parameter of shape (C,).
    beta : np.ndarray
        Shift parameter of shape (C,).
    eps : float
        Small constant for numerical stability.

    Returns:
    --------
    np.ndarray
        Normalized tensor of shape (B, C, H, W).
    """
    B, C, H, W = X.shape
    
    # Compute mean and variance across spatial dimensions (H, W)
    mean = X.mean(axis=(2, 3), keepdims=True)   # shape (B, C, 1, 1)
    var = X.var(axis=(2, 3), keepdims=True)     # shape (B, C, 1, 1)
    
    # Normalize
    X_hat = (X - mean) / np.sqrt(var + eps)
    
    # Apply scale (gamma) and shift (beta)
    # Reshape gamma and beta to broadcast correctly
    gamma = gamma.reshape(1, C, 1, 1)
    beta = beta.reshape(1, C, 1, 1)
    
    out = gamma * X_hat + beta
    return out

