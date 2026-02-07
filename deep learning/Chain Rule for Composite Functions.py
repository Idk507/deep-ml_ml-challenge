import numpy as np

def compute_chain_rule_gradient(functions: list[str], x: float) -> float:
    """
    Compute derivative of composite functions using chain rule.
    
    Args:
        functions: List of function names (applied right to left)
                  Available: 'square', 'sin', 'exp', 'log'
        x: Point at which to evaluate derivative
    
    Returns:
        Derivative value at x
    """
    
    # Function definitions
    func_map = {
        'square': lambda u: u ** 2,
        'sin': lambda u: np.sin(u),
        'exp': lambda u: np.exp(u),
        'log': lambda u: np.log(u)
    }
    
    # Derivative definitions
    deriv_map = {
        'square': lambda u: 2 * u,
        'sin': lambda u: np.cos(u),
        'exp': lambda u: np.exp(u),
        'log': lambda u: 1 / u
    }
    
    # Forward pass: compute intermediate values
    # Functions applied right to left, so reverse
    values = [x]
    current = x
    
    for func_name in reversed(functions):
        current = func_map[func_name](current)
        values.append(current)
    
    # Backward pass: multiply derivatives
    # Work from outermost to innermost function
    gradient = 1.0
    
    for i, func_name in enumerate(functions):
        # Input to this function is values[-(i+2)]
        input_val = values[-(i+2)]
        gradient *= deriv_map[func_name](input_val)
    
    return float(gradient)
