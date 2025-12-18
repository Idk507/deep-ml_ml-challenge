import numpy as np

def conv3d_forward_pass(
    input_volume: np.ndarray,
    kernel: np.ndarray,
    stride: tuple[int, int, int] = (1, 1, 1),
    padding: tuple[int, int, int] = (0, 0, 0)
) -> np.ndarray:
    """
    Perform 3D convolution forward pass.

    Args:
        input_volume: np.ndarray of shape (C, D, H, W)
        kernel: np.ndarray of shape (C, kD, kH, kW)
        stride: (stride_d, stride_h, stride_w)
        padding: (pad_d, pad_h, pad_w)

    Returns:
        output: np.ndarray of shape (1, D_out, H_out, W_out)
    """

    # ---------------------------
    # 1. Extract dimensions
    # ---------------------------
    C, D, H, W = input_volume.shape
    C_k, kD, kH, kW = kernel.shape

    assert C == C_k, "Kernel channels must match input channels"

    stride_d, stride_h, stride_w = stride
    pad_d, pad_h, pad_w = padding

    # ---------------------------
    # 2. Apply padding
    # ---------------------------
    padded_input = np.pad(
        input_volume,
        pad_width=(
            (0, 0),                  # channels
            (pad_d, pad_d),          # depth
            (pad_h, pad_h),          # height
            (pad_w, pad_w)           # width
        ),
        mode="constant",
        constant_values=0
    )

    _, D_p, H_p, W_p = padded_input.shape

    # ---------------------------
    # 3. Compute output dimensions
    # ---------------------------
    D_out = (D_p - kD) // stride_d + 1
    H_out = (H_p - kH) // stride_h + 1
    W_out = (W_p - kW) // stride_w + 1

    # Initialize output
    output = np.zeros((1, D_out, H_out, W_out), dtype=float)

    # ---------------------------
    # 4. Perform convolution
    # ---------------------------
    for d in range(D_out):
        for h in range(H_out):
            for w in range(W_out):

                d_start = d * stride_d
                h_start = h * stride_h
                w_start = w * stride_w

                # Extract 3D patch
                patch = padded_input[
                    :,
                    d_start:d_start + kD,
                    h_start:h_start + kH,
                    w_start:w_start + kW
                ]

                # Element-wise multiply and sum
                output[0, d, h, w] = np.sum(patch * kernel)

    return output
