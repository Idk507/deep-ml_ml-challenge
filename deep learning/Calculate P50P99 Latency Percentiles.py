import numpy as np

def calculate_latency_percentiles(latencies: list[float]) -> dict[str, float]:
    """
    Calculate P50, P95, and P99 latency percentiles.
    
    Args:
        latencies: List of latency measurements
    
    Returns:
        Dictionary with keys 'P50', 'P95', 'P99' containing
        the respective percentile values rounded to 4 decimal places
    """
    # Handle empty list
    if not latencies or len(latencies) == 0:
        return {'P50': 0.0, 'P95': 0.0, 'P99': 0.0}
    
    # Use numpy's percentile function with linear interpolation
    # interpolation='linear' matches the requirement
    p50 = np.percentile(latencies, 50, interpolation='linear')
    p95 = np.percentile(latencies, 95, interpolation='linear')
    p99 = np.percentile(latencies, 99, interpolation='linear')
    
    return {
        'P50': round(float(p50), 4),
        'P95': round(float(p95), 4),
        'P99': round(float(p99), 4)
    }
