"""
Implement a function to calculate the Expected Calibration Error (ECE) for evaluating how well a binary classification model's predicted probabilities are calibrated.

Given true binary labels (0 or 1) and predicted probabilities for the positive class, compute the ECE by:

Dividing the probability range [0, 1] into n_bins equal-width bins
For each non-empty bin, calculating the average predicted probability (confidence) and the actual accuracy (fraction of positive labels)
Computing the weighted average of the absolute differences between confidence and accuracy across all bins, where weights are the proportion of samples in each bin
The first bin should include both endpoints [0, 1/n_bins], while subsequent bins should exclude their lower boundary (lower, upper].

The function should return the ECE value rounded to 3 decimal places. Lower ECE values indicate better calibration, with 0 being perfectly calibrated.

Example:
Input:
y_true = [0, 0, 1, 1], y_prob = [0.8, 0.8, 0.8, 0.8], n_bins = 5
Output:
0.3
Reasoning:
All 4 predictions fall into the bin (0.6, 0.8]. The average confidence in this bin is 0.8, but the actual accuracy is 2/4 = 0.5 (only 2 out of 4 are positive). The absolute difference is |0.5 - 0.8| = 0.3. Since all samples are in this bin, the weight is 1.0, giving ECE = 1.0 * 0.3 = 0.3. This indicates an overconfident model.

Learn About topic
Expected Calibration Error (ECE)
Expected Calibration Error (ECE) is a widely-used metric for evaluating how well a model's predicted probabilities align with actual outcomes. A well-calibrated model should have predicted probabilities that match the true frequency of positive outcomes.

Why Calibration Matters
In many real-world applications, we need not just accurate predictions, but also reliable confidence estimates:

Medical diagnosis: A 90% confidence should mean 9 out of 10 similar cases are positive
Weather forecasting: "70% chance of rain" should correspond to rain 70% of the time
Risk assessment: Probability estimates directly affect decision-making
Mathematical Definition
ECE partitions predictions into 
M
M bins based on confidence levels and computes:

ECE
=
∑
m
=
1
M
∣
B
m
∣
n
⋅
∣
acc
(
B
m
)
−
conf
(
B
m
)
∣
ECE= 
m=1
∑
M
​
  
n
∣B 
m
​
 ∣
​
 ⋅∣acc(B 
m
​
 )−conf(B 
m
​
 )∣
Where:

B
m
B 
m
​
  is the set of samples whose predicted probability falls in bin 
m
m
∣
B
m
∣
∣B 
m
​
 ∣ is the number of samples in bin 
m
m
n
n is the total number of samples
acc
(
B
m
)
acc(B 
m
​
 ) is the accuracy (fraction of positive labels) in bin 
m
m
conf
(
B
m
)
conf(B 
m
​
 ) is the average predicted probability in bin 
m
m
Computing Accuracy and Confidence per Bin
For each bin 
B
m
B 
m
​
 :

acc
(
B
m
)
=
1
∣
B
m
∣
∑
i
∈
B
m
y
i
acc(B 
m
​
 )= 
∣B 
m
​
 ∣
1
​
  
i∈B 
m
​
 
∑
​
 y 
i
​
 
conf
(
B
m
)
=
1
∣
B
m
∣
∑
i
∈
B
m
p
^
i
conf(B 
m
​
 )= 
∣B 
m
​
 ∣
1
​
  
i∈B 
m
​
 
∑
​
  
p
^
​
  
i
​
 
Where 
y
i
∈
{
0
,
1
}
y 
i
​
 ∈{0,1} is the true label and 
p
^
i
p
^
​
  
i
​
  is the predicted probability.

Interpreting ECE Values
| ECE Value | Interpretation | |-----------|----------------| | 0.0 | Perfectly calibrated | | < 0.05 | Well calibrated | | 0.05 - 0.15 | Moderately calibrated | | > 0.15 | Poorly calibrated |

Types of Miscalibration
Overconfidence: When 
conf
(
B
m
)
>
acc
(
B
m
)
conf(B 
m
​
 )>acc(B 
m
​
 ), the model is overconfident - it predicts higher probabilities than the actual positive rate.

Underconfidence: When 
conf
(
B
m
)
<
acc
(
B
m
)
conf(B 
m
​
 )<acc(B 
m
​
 ), the model is underconfident - it predicts lower probabilities than warranted by the data.

Reliability Diagrams
ECE is often visualized using reliability diagrams, which plot:

X-axis: Mean predicted probability per bin (confidence)
Y-axis: Fraction of positives per bin (accuracy)
A perfectly calibrated model produces points along the diagonal line 
y
=
x
y=x.

Limitations
Bin sensitivity: ECE depends on the number and boundaries of bins
Sample size: Small bins may have unreliable statistics
Class imbalance: May need stratified versions for imbalanced datasets
Calibration Methods
When ECE is high, calibration techniques can improve probability estimates:

Platt Scaling: Fits a sigmoid function to transform outputs
Temperature Scaling: Divides logits by a learned temperature
Isotonic Regression: Non-parametric monotonic calibration

"""

import numpy as np

def expected_calibration_error(y_true, y_prob, n_bins=10):
    """
    Calculate the Expected Calibration Error (ECE).
    
    Args:
        y_true: List or array of true binary labels (0 or 1)
        y_prob: List or array of predicted probabilities for the positive class
        n_bins: Number of bins for grouping predictions (default: 10)
    
    Returns:
        float: ECE value rounded to 3 decimal places
    """
    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    
    n_samples = len(y_true)
    
    # Define bin boundaries: [0, 1/n_bins], (1/n_bins, 2/n_bins], ..., ((n_bins-1)/n_bins, 1]
    bin_edges = np.linspace(0, 1, n_bins + 1)
    
    ece = 0.0
    
    # Process each bin
    for i in range(n_bins):
        lower_bound = bin_edges[i]
        upper_bound = bin_edges[i + 1]
        
        # Determine which samples fall into this bin
        if i == 0:
            # First bin: [0, 1/n_bins] - includes both endpoints
            in_bin = (y_prob >= lower_bound) & (y_prob <= upper_bound)
        else:
            # Subsequent bins: (lower, upper] - excludes lower bound
            in_bin = (y_prob > lower_bound) & (y_prob <= upper_bound)
        
        # Count samples in this bin
        bin_size = np.sum(in_bin)
        
        # Skip empty bins
        if bin_size == 0:
            continue
        
        # Calculate confidence: average predicted probability in the bin
        confidence = np.mean(y_prob[in_bin])
        
        # Calculate accuracy: fraction of positive labels in the bin
        accuracy = np.mean(y_true[in_bin])
        
        # Calculate weight: proportion of samples in this bin
        weight = bin_size / n_samples
        
        # Add weighted absolute difference to ECE
        ece += weight * abs(accuracy - confidence)
    
    # Round to 3 decimal places
    return round(float(ece), 3)
