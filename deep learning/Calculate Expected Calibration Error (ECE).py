import numpy as np

def expected_calibration_error(y_true, y_prob, n_bins=10):
    """
    Calculate Expected Calibration Error (ECE) for binary classification.
    
    Args:
        y_true: Array of true binary labels (0 or 1)
        y_prob: Array of predicted probabilities for positive class
        n_bins: Number of bins to divide [0, 1] into
    
    Returns:
        ECE value rounded to 3 decimal places
    """
    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    
    n_samples = len(y_true)
    
    # Define bin boundaries
    # First bin: [0, 1/n_bins], subsequent bins: (i/n_bins, (i+1)/n_bins]
    bin_edges = np.linspace(0, 1, n_bins + 1)
    
    ece = 0.0
    
    # Process each bin
    for i in range(n_bins):
        lower_bound = bin_edges[i]
        upper_bound = bin_edges[i + 1]
        
        # Find samples in this bin
        if i == 0:
            # First bin includes both endpoints: [lower, upper]
            in_bin = (y_prob >= lower_bound) & (y_prob <= upper_bound)
        else:
            # Other bins exclude lower boundary: (lower, upper]
            in_bin = (y_prob > lower_bound) & (y_prob <= upper_bound)
        
        # Get samples in this bin
        bin_count = np.sum(in_bin)
        
        # Skip empty bins
        if bin_count == 0:
            continue
        
        # Calculate confidence (average predicted probability in bin)
        confidence = np.mean(y_prob[in_bin])
        
        # Calculate accuracy (fraction of positive labels in bin)
        accuracy = np.mean(y_true[in_bin])
        
        # Weight by proportion of samples in bin
        weight = bin_count / n_samples
        
        # Add weighted absolute difference to ECE
        ece += weight * np.abs(accuracy - confidence)
    
    return round(float(ece), 3)
