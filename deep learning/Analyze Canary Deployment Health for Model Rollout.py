"""
Analyze Canary Deployment Health for Model Rollout
Medium
MLOps

In production ML systems, canary deployments are a critical strategy for safely rolling out new model versions. A small percentage of traffic is routed to the new (canary) model while the majority continues to use the existing (baseline) model. By comparing their performance, you can decide whether to promote the canary to full production or roll back.

Given prediction results from both canary and baseline models, compute key comparison metrics to determine if the canary deployment is healthy.

Each result in both lists is a dictionary with:

'latency_ms': Response latency in milliseconds (float)
'prediction': The model's predicted value
'ground_truth': The actual correct value
Write a function analyze_canary_deployment(canary_results, baseline_results, accuracy_tolerance, latency_tolerance) that computes:

canary_accuracy: Fraction of correct predictions for canary model (0-1)
baseline_accuracy: Fraction of correct predictions for baseline model (0-1)
accuracy_change_pct: Relative change in accuracy as percentage
canary_avg_latency: Average latency of canary model (ms)
baseline_avg_latency: Average latency of baseline model (ms)
latency_change_pct: Relative change in latency as percentage
promote_recommended: Boolean - True if canary accuracy did not degrade beyond accuracy_tolerance AND latency did not increase beyond latency_tolerance
If either input list is empty, return an empty dictionary.

All numeric values should be rounded to 2 decimal places except accuracy values which should be rounded to 4 decimal places.

Example:
Input:
canary_results = [{'latency_ms': 45, 'prediction': 1, 'ground_truth': 1}, {'latency_ms': 50, 'prediction': 0, 'ground_truth': 0}, {'latency_ms': 48, 'prediction': 1, 'ground_truth': 1}, {'latency_ms': 52, 'prediction': 1, 'ground_truth': 0}, {'latency_ms': 47, 'prediction': 0, 'ground_truth': 0}], baseline_results = [{'latency_ms': 50, 'prediction': 1, 'ground_truth': 1}, {'latency_ms': 55, 'prediction': 0, 'ground_truth': 0}, {'latency_ms': 52, 'prediction': 1, 'ground_truth': 0}, {'latency_ms': 58, 'prediction': 0, 'ground_truth': 0}, {'latency_ms': 53, 'prediction': 1, 'ground_truth': 1}]
Output:
{'canary_accuracy': 0.8, 'baseline_accuracy': 0.8, 'accuracy_change_pct': 0.0, 'canary_avg_latency': 48.4, 'baseline_avg_latency': 53.6, 'latency_change_pct': -9.7, 'promote_recommended': True}
Reasoning:
Canary has 4/5 correct predictions (accuracy 0.8), baseline also has 4/5 (accuracy 0.8), so accuracy change is 0%. Canary average latency is (45+50+48+52+47)/5 = 48.4ms, baseline is (50+55+52+58+53)/5 = 53.6ms. Latency change is (48.4-53.6)/53.6 * 100 = -9.7% (improved). Since accuracy did not degrade and latency improved, promote_recommended is True.

Canary Deployments in MLOps
Canary deployment is a risk-mitigation strategy used when releasing new model versions to production. Instead of switching all traffic to the new model at once, you gradually route a small percentage (typically 1-10%) to the "canary" while monitoring its performance against the existing "baseline" model.

Key Metrics for Canary Analysis
1. Accuracy Comparison
Accuracy measures the fraction of correct predictions:

Accuracy
=
Correct Predictions
Total Predictions
Accuracy= 
Total Predictions
Correct Predictions
​
 

The relative accuracy change quantifies how the canary compares to baseline:

Accuracy Change %
=
A
c
a
n
a
r
y
−
A
b
a
s
e
l
i
n
e
A
b
a
s
e
l
i
n
e
×
100
Accuracy Change %= 
A 
baseline
​
 
A 
canary
​
 −A 
baseline
​
 
​
 ×100

where 
A
c
a
n
a
r
y
A 
canary
​
  and 
A
b
a
s
e
l
i
n
e
A 
baseline
​
  are the respective accuracies.

2. Latency Comparison
Average latency affects user experience:

L
ˉ
=
1
n
∑
i
=
1
n
L
i
L
ˉ
 = 
n
1
​
 ∑ 
i=1
n
​
 L 
i
​
 

The relative latency change:

Latency Change %
=
L
ˉ
c
a
n
a
r
y
−
L
ˉ
b
a
s
e
l
i
n
e
L
ˉ
b
a
s
e
l
i
n
e
×
100
Latency Change %= 
L
ˉ
  
baseline
​
 
L
ˉ
  
canary
​
 − 
L
ˉ
  
baseline
​
 
​
 ×100

Positive values indicate the canary is slower; negative values indicate improvement.

Promotion Decision Logic
The promotion decision uses tolerance thresholds:

Promote
=
(
Δ
a
c
c
≥
−
τ
a
c
c
)
∧
(
Δ
l
a
t
≤
τ
l
a
t
)
Promote=(Δ 
acc
​
 ≥−τ 
acc
​
 )∧(Δ 
lat
​
 ≤τ 
lat
​
 )

where:

Δ
a
c
c
Δ 
acc
​
  is the accuracy change percentage
Δ
l
a
t
Δ 
lat
​
  is the latency change percentage
τ
a
c
c
τ 
acc
​
  is the accuracy tolerance (e.g., 5%)
τ
l
a
t
τ 
lat
​
  is the latency tolerance (e.g., 10%)
Interpreting Results
| Scenario | Accuracy Change | Latency Change | Recommendation | |----------|----------------|----------------|----------------| | Ideal | Positive or zero | Negative or within tolerance | Promote | | Acceptable | Within tolerance | Within tolerance | Promote | | Risky | Beyond tolerance | Any | Do not promote | | Slow | Any | Beyond tolerance | Do not promote |

Best Practices
Statistical Significance: Ensure sufficient sample sizes before making decisions
Multiple Metrics: Consider additional metrics like error rates, p99 latency
Gradual Rollout: Increase canary traffic incrementally (1% -> 5% -> 25% -> 100%)
Automatic Rollback: Implement automated rollback if metrics degrade significantly
Segmented Analysis: Analyze performance across different user segments or request types
"""
def analyze_canary_deployment(canary_results: list, baseline_results: list, accuracy_tolerance: float = 0.05, latency_tolerance: float = 0.10) -> dict:
    """
    Analyze canary deployment health metrics for model rollout decision.
    
    Args:
        canary_results: list of prediction results from canary (new) model
                       Each dict has 'latency_ms', 'prediction', 'ground_truth'
        baseline_results: list of prediction results from baseline (existing) model
                         Each dict has 'latency_ms', 'prediction', 'ground_truth'
        accuracy_tolerance: max acceptable relative accuracy degradation (0.05 = 5%)
        latency_tolerance: max acceptable relative latency increase (0.10 = 10%)
    
    Returns:
        dict with canary/baseline metrics and promotion recommendation
    """
    # Handle empty inputs
    if not canary_results or not baseline_results:
        return {}
    
    # Calculate canary metrics
    canary_correct = sum(1 for r in canary_results if r['prediction'] == r['ground_truth'])
    canary_total = len(canary_results)
    canary_accuracy = canary_correct / canary_total
    
    canary_total_latency = sum(r['latency_ms'] for r in canary_results)
    canary_avg_latency = canary_total_latency / canary_total
    
    # Calculate baseline metrics
    baseline_correct = sum(1 for r in baseline_results if r['prediction'] == r['ground_truth'])
    baseline_total = len(baseline_results)
    baseline_accuracy = baseline_correct / baseline_total
    
    baseline_total_latency = sum(r['latency_ms'] for r in baseline_results)
    baseline_avg_latency = baseline_total_latency / baseline_total
    
    # Calculate relative changes as percentages
    # Accuracy change: (canary - baseline) / baseline * 100
    if baseline_accuracy > 0:
        accuracy_change_pct = ((canary_accuracy - baseline_accuracy) / baseline_accuracy) * 100
    else:
        # Edge case: if baseline accuracy is 0
        accuracy_change_pct = 0.0 if canary_accuracy == 0 else 100.0
    
    # Latency change: (canary - baseline) / baseline * 100
    if baseline_avg_latency > 0:
        latency_change_pct = ((canary_avg_latency - baseline_avg_latency) / baseline_avg_latency) * 100
    else:
        # Edge case: if baseline latency is 0 (shouldn't happen in practice)
        latency_change_pct = 0.0 if canary_avg_latency == 0 else 100.0
    
    # Determine promotion recommendation
    # Promote if:
    # 1. Accuracy did NOT degrade beyond tolerance (accuracy_change >= -accuracy_tolerance%)
    # 2. Latency did NOT increase beyond tolerance (latency_change <= latency_tolerance%)
    accuracy_tolerance_pct = accuracy_tolerance * 100  # Convert to percentage
    latency_tolerance_pct = latency_tolerance * 100    # Convert to percentage
    
    accuracy_acceptable = accuracy_change_pct >= -accuracy_tolerance_pct
    latency_acceptable = latency_change_pct <= latency_tolerance_pct
    
    promote_recommended = accuracy_acceptable and latency_acceptable
    
    # Return results with proper rounding
    return {
        'canary_accuracy': round(canary_accuracy, 4),
        'baseline_accuracy': round(baseline_accuracy, 4),
        'accuracy_change_pct': round(accuracy_change_pct, 2),
        'canary_avg_latency': round(canary_avg_latency, 2),
        'baseline_avg_latency': round(baseline_avg_latency, 2),
        'latency_change_pct': round(latency_change_pct, 2),
        'promote_recommended': promote_recommended
    }
