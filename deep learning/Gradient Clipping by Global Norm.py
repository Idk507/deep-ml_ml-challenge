import math
from typing import List

def clip_gradients_by_global_norm(gradients: List[List[float]], max_norm: float) -> List[List[float]]:
    """
    Clip gradients by global norm.

    Args:
        gradients: List of gradient arrays (each array corresponds to a parameter group).
        max_norm: Maximum allowed global norm.

    Returns:
        List of clipped gradient arrays maintaining the original structure.
    """
    # Step 1: Compute global L2 norm
    global_norm = math.sqrt(sum(g**2 for arr in gradients for g in arr))

    # Step 2: Compute clipping coefficient
    if global_norm > max_norm and global_norm > 0:
        scale = max_norm / global_norm
    else:
        scale = 1.0

    # Step 3: Scale gradients proportionally
    clipped_gradients = [[g * scale for g in arr] for arr in gradients]

    return clipped_gradients
