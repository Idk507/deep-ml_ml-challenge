"""
Implement a function to monitor changes in model prediction distributions between a reference (baseline) period and a current period. This is a critical MLOps task for detecting model drift in production.

Given two lists of prediction scores (probabilities between 0 and 1), compute the following monitoring metrics:

Mean Shift: The difference between the mean of current predictions and the mean of reference predictions
Standard Deviation Ratio: The ratio of current standard deviation to reference standard deviation
Jensen-Shannon Divergence: A symmetric measure of distribution similarity based on histogram comparison
Drift Detected: A boolean flag indicating if JS divergence exceeds 0.1 (significant drift threshold)
For the Jensen-Shannon divergence calculation:

Create histograms using n_bins equally spaced bins between 0 and 1
Apply Laplace smoothing to handle empty bins: P(bin) = (count + 1) / (total + n_bins)
Compute JS divergence as the average of KL divergences from each distribution to their mixture
Return a dictionary with keys 'mean_shift', 'std_ratio', 'js_divergence', and 'drift_detected'.

Example:
Input:
reference_preds = [0.1, 0.2, 0.3, 0.4, 0.5]
current_preds = [0.5, 0.6, 0.7, 0.8, 0.9]
n_bins = 5
Output:
{'mean_shift': 0.4, 'std_ratio': 1.0, 'js_divergence': 0.2939, 'drift_detected': True}
Reasoning:
The reference predictions have mean 0.3 and the current predictions have mean 0.7, giving a mean_shift of 0.4. Both have the same spread (std = 0.1414), so std_ratio = 1.0. The JS divergence of 0.2939 exceeds the 0.1 threshold, indicating significant distribution drift.

Learn About topic
Prediction Distribution Monitoring
In production ML systems, monitoring the distribution of model predictions is essential for detecting concept drift and data drift. When the distribution of predictions changes significantly from a baseline period, it may indicate that the model's behavior has degraded.

Key Monitoring Metrics
1. Mean Shift
The simplest metric tracks whether predictions are systematically higher or lower: 
Mean Shift
=
p
ˉ
c
u
r
r
e
n
t
−
p
ˉ
r
e
f
e
r
e
n
c
e
Mean Shift= 
p
ˉ
​
  
current
​
 − 
p
ˉ
​
  
reference
​
 

A large mean shift suggests the model is predicting differently on average.

2. Standard Deviation Ratio
This captures changes in prediction confidence spread: 
Std Ratio
=
σ
c
u
r
r
e
n
t
σ
r
e
f
e
r
e
n
c
e
Std Ratio= 
σ 
reference
​
 
σ 
current
​
 
​
 

Ratio > 1: Predictions are more spread out (less confident)
Ratio < 1: Predictions are more concentrated (potentially overconfident)
3. Jensen-Shannon Divergence
JS divergence is a symmetric measure of how different two probability distributions are: 
J
S
(
P
∥
Q
)
=
1
2
K
L
(
P
∥
M
)
+
1
2
K
L
(
Q
∥
M
)
JS(P∥Q)= 
2
1
​
 KL(P∥M)+ 
2
1
​
 KL(Q∥M)

where 
M
=
1
2
(
P
+
Q
)
M= 
2
1
​
 (P+Q) is the mixture distribution, and KL divergence is: 
K
L
(
P
∥
Q
)
=
∑
i
P
(
i
)
log
⁡
P
(
i
)
Q
(
i
)
KL(P∥Q)=∑ 
i
​
 P(i)log 
Q(i)
P(i)
​
 

Properties of JS Divergence:

Bounded: 
0
≤
J
S
≤
log
⁡
(
2
)
≈
0.693
0≤JS≤log(2)≈0.693
Symmetric: 
J
S
(
P
∥
Q
)
=
J
S
(
Q
∥
P
)
JS(P∥Q)=JS(Q∥P)
JS = 0 means identical distributions
Laplace Smoothing
To handle empty histogram bins (which would cause division by zero), we apply Laplace smoothing: 
P
(
b
i
n
)
=
c
o
u
n
t
+
1
t
o
t
a
l
+
n
b
i
n
s
P(bin)= 
total+n 
bins
​
 
count+1
​
 

This ensures all bins have non-zero probability.

Drift Detection Threshold
A common practice is to flag drift when 
J
S
>
0.1
JS>0.1, though this threshold should be tuned based on your specific application's sensitivity requirements.

Practical Considerations
Monitor continuously in production with sliding windows
Set up alerts when metrics exceed thresholds
Investigate root causes: data quality issues, upstream changes, or genuine concept drift
Consider retraining or model updates when drift is persistent


"""
import numpy as np

def monitor_prediction_distribution(reference_preds: list, current_preds: list, n_bins: int = 10) -> dict:
    """
    Monitor prediction distribution changes between reference and current predictions.
    
    Args:
        reference_preds: List of reference prediction scores (floats between 0 and 1)
        current_preds: List of current prediction scores (floats between 0 and 1)
        n_bins: Number of bins for histogram comparison
    
    Returns:
        Dictionary with keys: 'mean_shift', 'std_ratio', 'js_divergence', 'drift_detected'
    """
    # Convert to numpy arrays
    reference_preds = np.array(reference_preds)
    current_preds = np.array(current_preds)
    
    # 1. Calculate Mean Shift
    mean_reference = np.mean(reference_preds)
    mean_current = np.mean(current_preds)
    mean_shift = mean_current - mean_reference
    
    # 2. Calculate Standard Deviation Ratio
    std_reference = np.std(reference_preds, ddof=0)  # Population std
    std_current = np.std(current_preds, ddof=0)
    
    # Handle edge case where reference std is 0
    if std_reference == 0:
        std_ratio = 1.0 if std_current == 0 else float('inf')
    else:
        std_ratio = std_current / std_reference
    
    # 3. Calculate Jensen-Shannon Divergence
    # Create histograms with n_bins equally spaced bins between 0 and 1
    bins = np.linspace(0, 1, n_bins + 1)
    
    # Count samples in each bin
    ref_counts, _ = np.histogram(reference_preds, bins=bins)
    curr_counts, _ = np.histogram(current_preds, bins=bins)
    
    # Apply Laplace smoothing: P(bin) = (count + 1) / (total + n_bins)
    ref_total = len(reference_preds)
    curr_total = len(current_preds)
    
    P = (ref_counts + 1) / (ref_total + n_bins)
    Q = (curr_counts + 1) / (curr_total + n_bins)
    
    # Compute mixture distribution M = (P + Q) / 2
    M = (P + Q) / 2
    
    # Compute KL divergences
    # KL(P || M) = sum(P * log(P / M))
    # KL(Q || M) = sum(Q * log(Q / M))
    # Use np.log for natural logarithm
    
    # Avoid log(0) by using only non-zero entries (though Laplace smoothing ensures all > 0)
    kl_pm = np.sum(P * np.log(P / M))
    kl_qm = np.sum(Q * np.log(Q / M))
    
    # JS divergence = (KL(P||M) + KL(Q||M)) / 2
    js_divergence = (kl_pm + kl_qm) / 2
    
    # 4. Drift Detection
    drift_detected = js_divergence > 0.1
    
    # Return results rounded appropriately
    return {
        'mean_shift': round(float(mean_shift), 4),
        'std_ratio': round(float(std_ratio), 4),
        'js_divergence': round(float(js_divergence), 4),
        'drift_detected': bool(drift_detected)
    }
