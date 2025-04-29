"""
Write a Python function that calculates the determinant of a 4x4 matrix using Laplace's Expansion method. The function should take a single argument, a 4x4 matrix represented as a list of lists, and return the determinant of the matrix. The elements of the matrix can be integers or floating-point numbers. Implement the function recursively to handle the computation of determinants for the 3x3 minor matrices.

Example:
Input:
a = [[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]]
Output:
0
Reasoning:
Using Laplace's Expansion, the determinant of a 4x4 matrix is calculated by expanding it into minors and cofactors along any row or column. Given the symmetrical and linear nature of this specific matrix, its determinant is 0. The calculation for a generic 4x4 matrix involves more complex steps, breaking it down into the determinants of 3x3 matrices.


Determinant of a 4x4 Matrix using Laplace's Expansion
Laplace's Expansion, also known as cofactor expansion, is a method to calculate the determinant of a square matrix of any size. For a 4x4 matrix ( A ), this method involves expanding ( A ) into minors and cofactors along a chosen row or column.

Consider a 4x4 matrix ( A ):

A
=
(
a
11
a
12
a
13
a
14
a
21
a
22
a
23
a
24
a
31
a
32
a
33
a
34
a
41
a
42
a
43
a
44
)
A= 
​
  
a 
11
​
 
a 
21
​
 
a 
31
​
 
a 
41
​
 
​
  
a 
12
​
 
a 
22
​
 
a 
32
​
 
a 
42
​
 
​
  
a 
13
​
 
a 
23
​
 
a 
33
​
 
a 
43
​
 
​
  
a 
14
​
 
a 
24
​
 
a 
34
​
 
a 
44
​
 
​
  
​
 
The determinant of ( A ), ( \det(A) ), can be calculated by selecting any row or column (e.g., the first row) and using the formula that involves the elements of that row (or column), their corresponding cofactors, and the determinants of the 3x3 minor matrices obtained by removing the row and column of each element. This process is recursive, as calculating the determinants of the 3x3 matrices involves further expansions.

The expansion formula for the first row is:

det
⁡
(
A
)
=
a
11
C
11
−
a
12
C
12
+
a
13
C
13
−
a
14
C
14
det(A)=a 
11
​
 C 
11
​
 −a 
12
​
 C 
12
​
 +a 
13
​
 C 
13
​
 −a 
14
​
 C 
14
​
 
Explanation of Terms
Cofactor ( C_{ij} ): The cofactor of element ( a_{ij} ) is given by:
C
i
j
=
(
−
1
)
i
+
j
det
⁡
(
Minor of 
a
i
j
)
C 
ij
​
 =(−1) 
i+j
 det(Minor of a 
ij
​
 )
where the minor of ( a_{ij} ) is the determinant of the 3x3 matrix obtained by removing the ( i )th row and ( j )th column from ( A ).
Notes
The choice of row or column for expansion can be based on convenience, often selecting one with the most zeros to simplify calculations.
The process is recursive, breaking down the determinant calculation into smaller 3x3 determinants until reaching 2x2 determinants, which are simpler to compute.
This method is fundamental in linear algebra and provides a systematic approach for determinant calculation, especially for matrices larger than 3x3.

"""
def determinant_4x4(matrix: list[list[int|float]]) -> float:
	# Your recursive implementation here
	if len(matrix) == 1: return matrix[0][0]
    det = 0
    for i in range(len(matrix)):
        minor = [row[:i]+row[i+1:]for row in matrix[1:]]
        cofactor = ((-1)**i) * determinant_4x4(minor)
        det +=matrix[0][i]*cofactor 
    return det
