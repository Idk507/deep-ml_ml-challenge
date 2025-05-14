"""
Implement a function that computes the derivative of a polynomial term of the form c * x^n at a given point x, where c is a coefficient and n is the exponent. The function should return the value of the derivative, accounting for the coefficient in the power rule. This is useful for understanding how polynomials change at specific points in machine learning optimization problems.

Example:
Input:
poly_term_derivative(2.0, 3.0, 2.0)
Output:
12.0
Reasoning:
For the term 2 * x^2, the derivative is 2 * 2 * x^(2-1) = 4 * x. At x = 3, this evaluates to 4 * 3 = 12.0.

Learn About topic
Derivative of a Polynomial
A function's derivative is a way of quantifying the function's slope at a given point. It allows us to understand whether the function is increasing, decreasing or flat at specific input.

Taking the derivative of a polynomial at a single point follows a straight-forward rule. This question will show the rule and the edge case you should be on the look-out for.

Mathematical Definition
When calculating the slope of a function 
f
(
x
)
f(x), we usually require two points 
x
1
x 
1
​
  and 
x
2
x 
2
​
  and use the following formula:

f
(
x
2
)
−
f
(
x
1
)
x
2
−
x
1
x 
2
​
 −x 
1
​
 
f(x 
2
​
 )−f(x 
1
​
 )
​
 
A derivative generalizes that notion by calculating the slope of a function at a specific point. A derivative of a function 
f
(
x
)
f(x) is mathematically defined as:

d
f
(
x
)
d
x
=
lim
⁡
h
→
0
f
(
x
+
h
)
−
f
(
x
)
h
dx
df(x)
​
 = 
h→0
lim
​
  
h
f(x+h)−f(x)
​
 
Where:

x
x is the input to the function
h
h is the "step", which is equivalent to the difference 
x
2
−
x
1
x 
2
​
 −x 
1
​
  in the two-point slope-formula
Taking the limit as the step grows smaller and smaller, allow us to quantify the slope at a certain point, instead of having to consider two points as in other methods of finding the slope.

When taking the derivative of a polynomial function 
x
n
x 
n
 , where 
n
≠
0
n

=0, then the derivative is: 
n
x
n
−
1
nx 
n−1
 . In the special case where 
n
=
0
n=0 then the derivative is zero. This is because 
x
0
=
1
x 
0
 =1 if 
x
≠
0
x

=0.

A positive derivative indicates that the function is increasing in that point, a negative derivative indicates that the function is decreasing at that point. A derivative equal to zero indicates that the function is flat, which could potentially indicate a function's minimum or maximum.


"""
def poly_term_derivative(c: float, x: float, n: float) -> float:
    # Your code here
    if n == 0.0 : return 0.0
    return round(c*n*(x**(n-1)),4)
