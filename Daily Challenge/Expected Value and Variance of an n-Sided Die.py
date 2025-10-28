"""
Expected Value and Variance of an n-Sided Die
Write a Python function that computes the expected value and variance of a fair n-sided die roll. The die has faces numbered 1 through n, each equally likely. The function should return a tuple (expected_value, variance).

Example:
Input:
dice_statistics(6)
Output:
(3.5, 2.9167)
Reasoning:
For n=6, the expected value is (6+1)/2 = 3.5 and the variance is (6^2-1)/12 = 35/12 ≈ 2.9167.

Expected Value and Variance of an n-Sided Die
When rolling a fair n-sided die, each outcome (1 through n) is equally likely with probability 
1
n
n
1
​
 .

Expected Value
The expected value of a random variable 
X
X is given by:

E
[
X
]
=
∑
i
=
1
n
x
i
P
(
x
i
)
E[X]= 
i=1
∑
n
​
 x 
i
​
 P(x 
i
​
 )
For a fair n-sided die:

E
[
X
]
=
1
+
2
+
3
+
⋯
+
n
n
=
n
+
1
2
E[X]= 
n
1+2+3+⋯+n
​
 = 
2
n+1
​
 
Variance
Variance is computed as:

V
a
r
(
X
)
=
E
[
X
2
]
−
(
E
[
X
]
)
2
Var(X)=E[X 
2
 ]−(E[X]) 
2
 
First, compute 
E
[
X
2
]
E[X 
2
 ]:

E
[
X
2
]
=
1
2
+
2
2
+
3
2
+
⋯
+
n
2
n
=
(
n
+
1
)
(
2
n
+
1
)
6
E[X 
2
 ]= 
n
1 
2
 +2 
2
 +3 
2
 +⋯+n 
2
 
​
 = 
6
(n+1)(2n+1)
​
 
Thus:

V
a
r
(
X
)
=
(
n
+
1
)
(
2
n
+
1
)
6
−
(
n
+
1
2
)
2
Var(X)= 
6
(n+1)(2n+1)
​
 −( 
2
n+1
​
 ) 
2
 
Key Results
Expected Value: 
n
+
1
2
2
n+1
​
 
Variance: 
n
2
−
1
12
12
n 
2
 −1
​

"""
def dice_statistics(n: int) -> tuple[float, float]:
    """
    Compute the expected value and variance of a fair n-sided die roll.

    Args:
        n (int): Number of sides of the die

    Returns:
        tuple: (expected_value, variance)
    """
    expected_value = (n + 1) / 2
    variance = (n**2 - 1) / 12
    return (expected_value, round(variance, 4))
