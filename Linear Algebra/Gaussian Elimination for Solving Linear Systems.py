"""
Task: Implement the Gaussian Elimination Method
Your task is to implement the Gaussian Elimination method, which transforms a system of linear equations into an upper triangular matrix. This method can then be used to solve for the variables using backward substitution.

Write a function gaussian_elimination(A, b) that performs Gaussian Elimination with partial pivoting to solve the system (Ax = b).

The function should return the solution vector (x).

Example:
Input:
A = np.array([[2,8,4], [2,5,1], [4,10,-1]], dtype=float)
b = np.array([2,5,1], dtype=float)

print(gaussian_elimination(A, b))
Output:
[11.0, -4.0, 3.0]
Reasoning:
The Gaussian Elimination method transforms the system of equations into an upper triangular matrix and then uses backward substitution to solve for the variables.
Understanding Gaussian Elimination
Gaussian Elimination is used to replace matrix coefficients with a row-echelon form matrix, which can be more easily solved via backwards substitution.

Row-Echelon Form Criteria
Non-zero rows are above any rows of all zeros.
The leading entry of each non-zero row is to the right of the leading entry of the previous row.
The leading entry in any non-zero row is 1, and all entries below it in the same column are zeros.
Augmented Matrix
For a linear system 
A
x
=
b
Ax=b, an augmented matrix is a way of displaying all the numerical information in a linear system in a single matrix. This combines the coefficient matrix 
A
A and vector source 
b
b as follows:

(
a
11
a
21
a
31
b
1
a
12
a
22
a
32
b
2
a
31
a
32
a
33
b
3
)
​
  
a 
11
​
 
a 
12
​
 
a 
31
​
 
​
  
a 
21
​
 
a 
22
​
 
a 
32
​
 
​
  
a 
31
​
 
a 
32
​
 
a 
33
​
 
​
  
b 
1
​
 
b 
2
​
 
b 
3
​
 
​
  
​
 
Partial Pivoting
In linear algebra, diagonal elements of a matrix are referred to as the "pivot". To solve a linear system, the diagonal is used as a divisor for other elements within the matrix. This means that Gaussian Elimination will fail if there is a zero pivot.

In this case, pivoting is used to interchange rows, ensuring a non-zero pivot. Specifically, partial pivoting looks at all other rows in the current column to find the row with the highest absolute value. This row is then interchanged with the current row. This not only increases the numerical stability of the solution, but also reduces round-off errors caused by dividing by small entries.

Gaussian Elimination Mathematical Formulation
Gaussian Elimination:

For 
k
=
1
k=1 to 
number of rows
−
1
number of rows−1:
Apply partial pivoting to the current row.
For 
i
=
k
+
1
i=k+1 to 
number of rows
number of rows:
m
i
k
=
a
i
k
a
k
k
m 
ik
​
 = 
a 
kk
​
 
a 
ik
​
 
​
 
For 
j
=
k
j=k to 
number of columns
number of columns:
a
i
j
=
a
i
j
−
m
i
k
×
a
k
j
a 
ij
​
 =a 
ij
​
 −m 
ik
​
 ×a 
kj
​
 
b
i
=
b
i
−
m
i
k
×
b
k
b 
i
​
 =b 
i
​
 −m 
ik
​
 ×b 
k
​
 
Backwards Substitution:

For 
k
=
number of rows
k=number of rows to 
1
1:
For 
i
=
number of columns
−
1
i=number of columns−1 to 
1
1:
b
k
=
b
k
−
a
k
i
×
b
i
b 
k
​
 =b 
k
​
 −a 
ki
​
 ×b 
i
​
 
b
k
=
b
k
a
k
k
b 
k
​
 = 
a 
kk
​
 
b 
k
​
 
​
 
Example Calculation
Letâs solve the system of equations:

A
=
(
2
8
4
5
5
1
4
10
−
1
)
and
b
=
(
2
5
1
)
A= 
​
  
2
5
4
​
  
8
5
10
​
  
4
1
−1
​
  
​
 andb= 
​
  
2
5
1
​
  
​
 
Apply partial pivoting to increase the magnitude of the pivot. For 
A
11
A 
11
​
 , calculate the factor for the elimination of 
A
12
A 
12
​
 :
m
12
=
A
12
A
11
=
2
5
=
0.4
m 
12
​
 = 
A 
11
​
 
A 
12
​
 
​
 = 
5
2
​
 =0.4
Apply this scaling to row 1 and subtract this from row 2, eliminating 
A
12
A 
12
​
 :
A
=
(
5
5
1
0
6
3.6
4
10
−
1
)
and
b
=
(
5
0
1
)
A= 
​
  
5
0
4
​
  
5
6
10
​
  
1
3.6
−1
​
  
​
 andb= 
​
  
5
0
1
​
  
​
 
After the full Gaussian Elimination process has been applied to 
A
A and 
b
b, we get the following:

A
=
(
5
5
1
0
6
3.6
0
0
−
5.4
)
and
b
=
(
5
0
3
)
A= 
​
  
5
0
0
​
  
5
6
0
​
  
1
3.6
−5.4
​
  
​
 andb= 
​
  
5
0
3
​
  
​
 
To calculate 
x
x, we apply backward substitution by substituting in the currently solved values and dividing by the pivot. This gives the following for the first iteration:

x
3
=
b
3
A
33
=
3
−
5.4
=
−
0.56
x 
3
​
 = 
A 
33
​
 
b 
3
​
 
​
 = 
−5.4
3
​
 =−0.56
This process can be repeated iteratively for all rows to solve the linear system, substituting in the solved values for the rows below.

Applications
Gaussian Elimination and linear solvers have a wide range of real-world applications, including their use in:

Machine learning
Computational fluid dynamics
3D graphics
"""
