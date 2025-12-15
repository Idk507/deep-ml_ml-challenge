import numpy as np

FP8_E4M3_MAX = 448.0
EPS = 1e-12

def fp8_block_quantize(
    tensor: np.ndarray,
    block_size: int = 128
) -> tuple[np.ndarray, np.ndarray]:
    """
    Quantize a tensor to FP8-E4M3 format using block-wise scaling.

    Args:
        tensor: Input tensor of shape (N,) where N is divisible by block_size
        block_size: Number of elements per quantization block

    Returns:
        quantized: Quantized values of shape (N,), clipped to [-448, 448]
        scales: Per-block scale factors of shape (N // block_size,)
    """
    if tensor.ndim != 1:
        raise ValueError("tensor must be 1D (shape (N,)).")
    N = tensor.shape[0]
    if N % block_size != 0:
        raise ValueError("N must be divisible by block_size.")
    
    num_blocks = N // block_size
    quantized = np.empty_like(tensor, dtype=np.int32)  # int32 to hold range safely
    scales = np.empty(num_blocks, dtype=np.float32)

    for b in range(num_blocks):
        start = b * block_size
        end = start + block_size
        block = tensor[start:end]

        max_abs = np.max(np.abs(block))
        scale = (max_abs + EPS) / FP8_E4M3_MAX  # dequantization scale (per spec)
        scales[b] = scale

        q = np.round(block / scale)  # quantization: divide by scale
        q = np.clip(q, -FP8_E4M3_MAX, FP8_E4M3_MAX)
        quantized[start:end] = q.astype(np.int32)

    return quantized, scales


def fp8_block_dequantize(
    quantized: np.ndarray,
    scales: np.ndarray,
    block_size: int = 128
) -> np.ndarray:
    """
    Dequantize FP8-E4M3 values back to full precision.

    Args:
        quantized: Quantized values of shape (N,)
        scales: Per-block scale factors of shape (N // block_size,)
        block_size: Number of elements per quantization block

    Returns:
        Dequantized tensor of shape (N,)
    """
    if quantized.ndim != 1:
        raise ValueError("quantized must be 1D (shape (N,)).")
    N = quantized.shape[0]
    if N % block_size != 0:
        raise ValueError("N must be divisible by block_size.")
    if scales.shape[0] != N // block_size:
        raise ValueError("scales length must be N // block_size.")

    dequant = np.empty_like(quantized, dtype=np.float32)
    num_blocks = N // block_size

    for b in range(num_blocks):
        start = b * block_size
        end = start + block_size
        scale = scales[b]
        dequant[start:end] = quantized[start:end].astype(np.float32) * scale

    return dequant
