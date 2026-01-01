"""
Implement a two-sample independent t-test (Welch's t-test) to determine if two samples have significantly different means. The test should compute the t-statistic, p-value using the Welch-Satterthwaite degrees of freedom, make a decision to reject or fail to reject the null hypothesis, and calculate Cohen's d effect size. Welch's t-test does not assume equal variances between groups. Given two samples and a significance level alpha, return a dictionary with the test results.

Example:
Input:
sample1=[12, 14, 13, 15, 14], sample2=[8, 9, 10, 9, 11], alpha=0.05
Output:
{'t_statistic': 5.8244, 'p_value': 0.000394, 'degrees_of_freedom': 8.0, 'reject_null': True, 'cohens_d': 3.6836}
Reasoning:
Mean₁=13.6, Mean₂=9.4, Var₁=1.3, Var₂=1.3. SE = sqrt(1.3/5 + 1.3/5) = 0.721. t = (13.6-9.4)/0.721 = 5.824. df = 8.0. Two-tailed p = 0.000394. Since p < 0.05, reject null hypothesis. Cohen's d = 4.2/1.14 = 3.684 (very large effect).

Learn About topic
Two-Sample T-Test (Welch's Test)
The two-sample t-test determines whether two independent groups have significantly different means.

Hypothesis Framework
Null hypothesis (
H
0
H 
0
​
 ): 
μ
1
=
μ
2
μ 
1
​
 =μ 
2
​
  (means are equal)

Alternative hypothesis (
H
a
H 
a
​
 ): 
μ
1
≠
μ
2
μ 
1
​
 

=μ 
2
​
  (means differ)

Decision: If 
p
<
α
p<α, reject 
H
0
H 
0
​
 

Welch's T-Test Formulas
T-statistic:

t
=
X
ˉ
1
−
X
ˉ
2
s
1
2
n
1
+
s
2
2
n
2
t= 
n 
1
​
 
s 
1
2
​
 
​
 + 
n 
2
​
 
s 
2
2
​
 
​
 
​
 
X
ˉ
  
1
​
 − 
X
ˉ
  
2
​
 
​
 
Degrees of freedom (Welch-Satterthwaite):

df
=
(
s
1
2
n
1
+
s
2
2
n
2
)
2
(
s
1
2
/
n
1
)
2
n
1
−
1
+
(
s
2
2
/
n
2
)
2
n
2
−
1
df= 
n 
1
​
 −1
(s 
1
2
​
 /n 
1
​
 ) 
2
 
​
 + 
n 
2
​
 −1
(s 
2
2
​
 /n 
2
​
 ) 
2
 
​
 
( 
n 
1
​
 
s 
1
2
​
 
​
 + 
n 
2
​
 
s 
2
2
​
 
​
 ) 
2
 
​
 
P-value (two-tailed):

p
=
2
×
(
1
−
CDF
t
(
∣
t
∣
,
df
)
)
p=2×(1−CDF 
t
​
 (∣t∣,df))
Cohen's d (effect size):

d
=
X
ˉ
1
−
X
ˉ
2
s
pooled
,
s
pooled
=
(
n
1
−
1
)
s
1
2
+
(
n
2
−
1
)
s
2
2
n
1
+
n
2
−
2
d= 
s 
pooled
​
 
X
ˉ
  
1
​
 − 
X
ˉ
  
2
​
 
​
 ,s 
pooled
​
 = 
n 
1
​
 +n 
2
​
 −2
(n 
1
​
 −1)s 
1
2
​
 +(n 
2
​
 −1)s 
2
2
​
 
​
 
​
 
Algorithm
Step 1: Calculate means and variances

X
ˉ
i
=
1
n
i
∑
x
i
,
s
i
2
=
1
n
i
−
1
∑
(
x
i
−
X
ˉ
i
)
2
X
ˉ
  
i
​
 = 
n 
i
​
 
1
​
 ∑x 
i
​
 ,s 
i
2
​
 = 
n 
i
​
 −1
1
​
 ∑(x 
i
​
 − 
X
ˉ
  
i
​
 ) 
2
 
Step 2: Calculate standard error

SE
=
s
1
2
n
1
+
s
2
2
n
2
SE= 
n 
1
​
 
s 
1
2
​
 
​
 + 
n 
2
​
 
s 
2
2
​
 
​
 
​
 
Step 3: Calculate t-statistic

t
=
X
ˉ
1
−
X
ˉ
2
SE
t= 
SE
X
ˉ
  
1
​
 − 
X
ˉ
  
2
​
 
​
 
Step 4: Calculate df using Welch-Satterthwaite equation

Step 5: Calculate p-value from t-distribution

Step 6: Make decision and compute Cohen's d

Example Calculation
Given: Sample 1 = 
[
12
,
14
,
13
,
15
,
14
]
[12,14,13,15,14], Sample 2 = 
[
8
,
9
,
10
,
9
,
11
]
[8,9,10,9,11], 
α
=
0.05
α=0.05

Step 1: Statistics

n
1
=
5
,
n
2
=
5
n 
1
​
 =5,n 
2
​
 =5
X
ˉ
1
=
13.6
,
X
ˉ
2
=
9.4
X
ˉ
  
1
​
 =13.6, 
X
ˉ
  
2
​
 =9.4
s
1
2
=
1.3
,
s
2
2
=
1.3
s 
1
2
​
 =1.3,s 
2
2
​
 =1.3
Step 2: Standard error

SE
=
1.3
5
+
1.3
5
=
0.721
SE= 
5
1.3
​
 + 
5
1.3
​
 
​
 =0.721
Step 3: T-statistic

t
=
13.6
−
9.4
0.721
=
5.824
t= 
0.721
13.6−9.4
​
 =5.824
Step 4: Degrees of freedom

df
=
8.0
df=8.0
Step 5: P-value

p
=
2
×
P
(
T
>
5.824
)
=
0.000394
p=2×P(T>5.824)=0.000394
Step 6: Decision and effect size

Since 
p
<
0.05
p<0.05, reject 
H
0
H 
0
​
 
Cohen's d: 
d
=
4.2
/
1.14
=
3.68
d=4.2/1.14=3.68 (very large effect)
Conclusion: Strong evidence that groups differ (t(8) = 5.82, p < 0.001, d = 3.68)

Cohen's d Interpretation
0.2 = Small effect
0.5 = Medium effect
0.8 = Large effect
> 1.2 = Very large effect
Key insight: Statistical significance (p-value) â  practical significance (effect size). Large samples can detect tiny, meaningless differences.

Welch's vs Student's T-Test
Welch's (recommended):

No equal variance assumption
More robust
Adjusted df
Student's (traditional):

Assumes equal variances
Uses pooled variance
df = 
n
1
+
n
2
−
2
n 
1
​
 +n 
2
​
 −2
Rule: Use Welch's unless you have strong evidence of equal variances.


Assumptions
Independence: Samples are independent
Normality: Approximately normal (or n > 30 by CLT)
Random sampling: Samples randomly selected
Interpreting Output
{
    't_statistic': 5.8244,
    'p_value': 0.000394,
    'degrees_of_freedom': 8.0,
    'reject_null': True,
    'cohens_d': 3.6836
}
What it means:

t = 5.82: Difference is 5.82 standard errors from zero
p = 0.0004: Only 0.04% chance if groups truly equal
reject = True: Strong evidence groups differ
d = 3.68: Very large practical effect
Write-up: "A Welch's t-test showed significant difference (t(8) = 5.82, p < 0.001, d = 3.68), with Group 1 (M = 13.6) higher than Group 2 (M = 9.4)."

Common Pitfalls
Multiple comparisons: Use ANOVA for >2 groups
Confusing correlation with causation: Difference â  cause
Ignoring effect size: p-value alone insufficient
Wrong test: Use paired t-test for dependent samples
Implementation Tips
Variance: Use Bessel's correction (n-1)

var1 = np.var(sample1, ddof=1)
P-value: Two-tailed requires multiplying by 2

p = 2 * (1 - stats.t.cdf(abs(t), df))


"""
import numpy as np
from scipy import stats

def two_sample_t_test(sample1: list[float], sample2: list[float], alpha: float = 0.05) -> dict:
    """
    Perform a two-sample independent t-test (Welch's t-test).
    
    Args:
        sample1: First sample data
        sample2: Second sample data
        alpha: Significance level (default 0.05)
    
    Returns:
        Dictionary containing:
        - t_statistic: The calculated t-statistic
        - p_value: Two-tailed p-value
        - degrees_of_freedom: Degrees of freedom (Welch-Satterthwaite)
        - reject_null: Boolean, whether to reject null hypothesis
        - cohens_d: Effect size (Cohen's d)
    """
    # Convert to numpy arrays
    sample1 = np.array(sample1)
    sample2 = np.array(sample2)
    
    # Sample sizes
    n1, n2 = len(sample1), len(sample2)
    
    # Means
    mean1, mean2 = np.mean(sample1), np.mean(sample2)
    
    # Variances (Bessel's correction: ddof=1)
    var1, var2 = np.var(sample1, ddof=1), np.var(sample2, ddof=1)
    
    # Standard error
    se = np.sqrt(var1/n1 + var2/n2)
    
    # T-statistic
    t_stat = (mean1 - mean2) / se
    
    # Welch-Satterthwaite degrees of freedom
    df_num = (var1/n1 + var2/n2)**2
    df_den = (var1**2 / (n1**2 * (n1-1))) + (var2**2 / (n2**2 * (n2-1)))
    df = df_num / df_den
    
    # Two-tailed p-value
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))
    
    # Decision
    reject_null = p_value < alpha
    
    # Cohen's d (using pooled standard deviation)
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    cohens_d = (mean1 - mean2) / pooled_std
    
    return {
        't_statistic': round(t_stat, 4),
        'p_value': round(p_value, 6),
        'degrees_of_freedom': round(df, 2),
        'reject_null': reject_null,
        'cohens_d': round(cohens_d, 4)
    }
