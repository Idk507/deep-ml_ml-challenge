"""
Confidence Interval for Population Mean
Medium
Statistics

Implement a function to calculate a confidence interval for a population mean using the t-distribution. Given sample data and a confidence level (e.g., 0.95 for 95%), compute the sample mean, standard error, margin of error, and the lower and upper bounds of the confidence interval. The function should return a dictionary with all relevant statistics.

Example:
Input:
data=[10, 12, 11, 13, 14, 10, 12, 11], confidence_level=0.95
Output:
{'mean': 11.625, 'standard_error': 0.4978, 'margin_of_error': 1.177, 'lower_bound': 10.448, 'upper_bound': 12.802, 'confidence_level': 0.95}
Reasoning:
n=8, mean=11.625, s=1.408. SE = 1.408/√8 = 0.498. With df=7 and 95% confidence, t-critical = 2.365. ME = 2.365 × 0.498 = 1.177. CI = [11.625 - 1.177, 11.625 + 1.177] = [10.448, 12.802]. We are 95% confident the true mean is in this range.

Learn About topic
Understanding Confidence Intervals
A confidence interval provides a range of plausible values for an unknown population parameter. Instead of giving a single estimate, we acknowledge uncertainty by giving an interval.

The Core Idea
When we collect a sample and calculate the mean, that sample mean is our point estimate of the true population mean. But we know:

Different samples give different means
Our sample mean probably isn't exactly the true population mean
We need to quantify our uncertainty
A confidence interval says: "We are X% confident the true population mean lies within this range."

What Does "95% Confident" Mean?
Common misconception: "There's a 95% chance the true mean is in this interval."

Correct interpretation: "If we repeated this sampling procedure many times, 95% of the resulting confidence intervals would contain the true population mean."

The true mean is fixed (but unknown). The confidence interval is random (it varies with each sample). Our confidence is in the procedure, not any single interval.

Mathematical Framework
For a population mean 
μ
μ, the confidence interval is:

x
ˉ
±
t
α
/
2
,
d
f
×
SE
x
ˉ
 ±t 
α/2,df
​
 ×SE
Where:

x
ˉ
x
ˉ
  = sample mean (point estimate)
t
α
/
2
,
d
f
t 
α/2,df
​
  = critical value from t-distribution
SE
SE = standard error of the mean
α
α = 1 - confidence level
d
f
df = degrees of freedom = n - 1
Components Explained
1. Sample Mean (
x
ˉ
x
ˉ
 )

x
ˉ
=
1
n
∑
i
=
1
n
x
i
x
ˉ
 = 
n
1
​
  
i=1
∑
n
​
 x 
i
​
 
This is our best point estimate of the population mean.

2. Standard Error (SE)

SE
=
s
n
SE= 
n
​
 
s
​
 
Where 
s
s is the sample standard deviation:

s
=
1
n
−
1
∑
i
=
1
n
(
x
i
−
x
ˉ
)
2
s= 
n−1
1
​
  
i=1
∑
n
​
 (x 
i
​
 − 
x
ˉ
 ) 
2
 
​
 
The SE measures how much the sample mean varies from sample to sample. It decreases with sample size (larger samples give more precise estimates).

3. Critical Value (
t
α
/
2
,
d
f
t 
α/2,df
​
 )

This comes from the t-distribution with 
d
f
=
n
−
1
df=n−1 degrees of freedom.

For a 95% confidence interval:

α
=
1
−
0.95
=
0.05
α=1−0.95=0.05
α
/
2
=
0.025
α/2=0.025 (split between two tails)
We find 
t
t such that 
P
(
T
>
t
)
=
0.025
P(T>t)=0.025
The t-distribution accounts for the extra uncertainty when we estimate the population standard deviation from our sample.

4. Margin of Error (ME)

ME
=
t
α
/
2
,
d
f
×
SE
ME=t 
α/2,df
​
 ×SE
This is how much we add and subtract from the sample mean to get the interval bounds.

5. Confidence Interval Bounds

Lower bound
=
x
ˉ
−
ME
Upper bound
=
x
ˉ
+
ME
Lower bound
Upper bound
​
  
= 
x
ˉ
 −ME
= 
x
ˉ
 +ME
​
  
​
 
Step-by-Step Logic
Given: Sample data 
[
x
1
,
x
2
,
…
,
x
n
]
[x 
1
​
 ,x 
2
​
 ,…,x 
n
​
 ] and confidence level (e.g., 0.95)

Step 1: Calculate sample mean 
x
ˉ
x
ˉ
 

Step 2: Calculate sample standard deviation 
s
s (use 
n
−
1
n−1 in denominator for unbiased estimate)

Step 3: Calculate standard error

SE
=
s
n
SE= 
n
​
 
s
​
 
Step 4: Determine degrees of freedom

d
f
=
n
−
1
df=n−1
Step 5: Find critical value from t-distribution

Calculate 
α
=
1
−
confidence level
α=1−confidence level
Look up 
t
α
/
2
,
d
f
t 
α/2,df
​
  (the value where 
P
(
T
>
t
)
=
α
/
2
P(T>t)=α/2)
Step 6: Calculate margin of error

ME
=
t
α
/
2
,
d
f
×
SE
ME=t 
α/2,df
​
 ×SE
Step 7: Construct interval

CI
=
[
x
ˉ
−
ME
,
x
ˉ
+
ME
]
CI=[ 
x
ˉ
 −ME, 
x
ˉ
 +ME]
Example Walkthrough
Sample: 
[
10
,
12
,
11
,
13
,
14
,
10
,
12
,
11
]
[10,12,11,13,14,10,12,11], 95% confidence

Step 1: Mean

x
ˉ
=
10
+
12
+
11
+
13
+
14
+
10
+
12
+
11
8
=
11.625
x
ˉ
 = 
8
10+12+11+13+14+10+12+11
​
 =11.625
Step 2: Standard deviation

s
=
1.408
s=1.408
Step 3: Standard error

SE
=
1.408
8
=
0.498
SE= 
8
​
 
1.408
​
 =0.498
Step 4: Degrees of freedom

d
f
=
8
−
1
=
7
df=8−1=7
Step 5: Critical value (95% CI, df=7)

t
0.025
,
7
=
2.365
t 
0.025,7
​
 =2.365
Step 6: Margin of error

ME
=
2.365
×
0.498
=
1.177
ME=2.365×0.498=1.177
Step 7: Confidence interval

Lower
=
11.625
−
1.177
=
10.448
Upper
=
11.625
+
1.177
=
12.802
Lower
Upper
​
  
=11.625−1.177=10.448
=11.625+1.177=12.802
​
  
​
 
Result: 95% CI = [10.448, 12.802]

Interpretation: We are 95% confident that the true population mean lies between 10.448 and 12.802.

Key Relationships
1. Confidence Level vs Width

Higher confidence â Wider interval

80% CI: Narrow (t-critical â 1.28)
90% CI: Medium (t-critical â 1.65)
95% CI: Wide (t-critical â 1.96)
99% CI: Very wide (t-critical â 2.58)
Why? To be more confident we've captured the true mean, we need a wider net.

2. Sample Size vs Width

Larger sample â Narrower interval

SE
=
s
n
SE= 
n
​
 
s
​
 
As 
n
n increases, SE decreases, so ME decreases, making the interval narrower.

3. Variability vs Width

Higher variance â Wider interval

If data is more spread out (larger 
s
s), the SE is larger, leading to a wider interval.

T-Distribution vs Z-Distribution
Use t-distribution when:

Sample size is small (n < 30)
Population standard deviation is unknown (typical case)
Use z-distribution when:

Sample size is large (n â¥ 30)
Population standard deviation is known (rare)
For large samples, t-distribution approaches z-distribution. We use t-distribution here because it's more general and appropriate when we estimate 
σ
σ from sample data.

Common Applications
1. Quality Control: "Average widget weight is 50.2g, 95% CI [49.8, 50.6]"

2. Clinical Trials: "Mean reduction in blood pressure is 12 mmHg, 95% CI [8, 16]"

3. A/B Testing: "Average conversion rate difference is 2.1%, 95% CI [0.5%, 3.7%]" (significant since 0 not in interval)

4. Survey Research: "Mean satisfaction score is 7.2, 95% CI [6.8, 7.6]"

Important Concepts
Point Estimate vs Interval Estimate:

Point estimate: Single value (e.g., mean = 11.625)
Interval estimate: Range of values (e.g., [10.448, 12.802])
Interval conveys uncertainty
Precision:

Narrow CI = High precision (good)
Wide CI = Low precision (more uncertainty)
Precision improves with larger sample size
Accuracy:

Whether interval contains true value
We can't know this for any single interval
But our procedure is correct X% of the time
Trade-offs:

More confidence â Less precision (wider interval)
More data â More precision (narrower interval)
Choose confidence level based on context (95% is standard)
Critical Values Reference
For 95% confidence, t-critical values:

n=5 (df=4): t = 2.776
n=10 (df=9): t = 2.262
n=20 (df=19): t = 2.093
n=30 (df=29): t = 2.045
n=â: t â 1.96 (z-value)
Notice: As sample size increases, t-critical approaches the z-critical value of 1.96.

Assumptions
Random sampling: Data is randomly sampled from population
Independence: Observations are independent
Normality: Data is approximately normally distributed
Less critical with large samples (n > 30) due to Central Limit Theorem
More critical with small samples
If assumptions are violated, the confidence level may not be accurate.


"""

import numpy as np
from scipy import stats

def confidence_interval(data: list[float], confidence_level: float = 0.95) -> dict:
    """
    Calculate confidence interval for population mean using t-distribution.
    
    Args:
        data: Sample data (list of floats)
        confidence_level: Confidence level (default 0.95)
    
    Returns:
        Dictionary containing:
        - mean: Sample mean (point estimate)
        - standard_error: Standard error of the mean
        - margin_of_error: Margin of error
        - lower_bound: Lower bound of CI
        - upper_bound: Upper bound of CI
        - confidence_level: Confidence level used
    """
    # Convert to numpy array
    data = np.array(data)
    n = len(data)
    
    # Step 1: Sample mean
    mean = np.mean(data)
    
    # Step 2: Sample standard deviation (unbiased, ddof=1)
    sample_std = np.std(data, ddof=1)
    
    # Step 3: Standard error
    standard_error = sample_std / np.sqrt(n)
    
    # Step 4: Degrees of freedom
    df = n - 1
    
    # Step 5: Critical value from t-distribution
    alpha = 1 - confidence_level
    t_critical = stats.t.ppf(1 - alpha/2, df)
    
    # Step 6: Margin of error
    margin_of_error = t_critical * standard_error
    
    # Step 7: Confidence interval bounds
    lower_bound = mean - margin_of_error
    upper_bound = mean + margin_of_error
    
    return {
        'mean': round(mean, 4),
        'standard_error': round(standard_error, 4),
        'margin_of_error': round(margin_of_error, 4),
        'lower_bound': round(lower_bound, 4),
        'upper_bound': round(upper_bound, 4),
        'confidence_level': confidence_level
    }
