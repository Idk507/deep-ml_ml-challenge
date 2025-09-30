"""
Compute the mean End-Point Error (EPE) between a predicted optical flow field and its ground-truth, optionally using an occlusion/validity mask and an outlier clip (max_flow). Your function should ignore invalid values (NaN/Â±Inf), support Python lists or NumPy arrays with shape (H, W, 2), and return -1 if inputs are malformed or no valid pixels remain.

Example:
Input:
pred = [[[1,0],[0,1]], [[-1,0],[0,-1]]]; gt = [[[0,0],[0,0]], [[0,0],[0,0]]]
print(round(flow_epe(pred, gt), 4))
Output:
1.0
Reasoning:
Each of the four pixels has EPE = 1, so the mean is 1.0.

"""
from typing import Optional, Union

try:
    import numpy as np
except Exception:
    np = None

ArrayLike = Union[list, "np.ndarray"]

def flow_epe(pred: ArrayLike,
             gt: ArrayLike,
             mask: Optional[ArrayLike] = None,
             max_flow: Optional[float] = None) -> float:
    if np is None:
        return -1

    try:
        pred = np.asarray(pred, dtype=np.float32)
        gt = np.asarray(gt, dtype=np.float32)
        if pred.shape != gt.shape or pred.ndim != 3 or pred.shape[2] != 2:
            return -1

        # Compute EPE per pixel
        epe = np.linalg.norm(pred - gt, axis=2)

        # Create validity mask
        valid = np.isfinite(epe)
        if mask is not None:
            mask = np.asarray(mask, dtype=bool)
            if mask.shape != epe.shape:
                try:
                    mask = np.broadcast_to(mask, epe.shape)
                except Exception:
                    return -1
            valid &= mask

        # Apply clipping
        if max_flow is not None:
            epe = np.minimum(epe, max_flow)

        # Filter valid EPEs
        valid_epe = epe[valid]
        if valid_epe.size == 0:
            return -1

        return float(np.mean(valid_epe))

    except Exception:
        return -1
