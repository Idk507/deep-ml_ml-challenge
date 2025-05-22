"""
Write a Python function that computes the element-wise sum of two vectors. The function should return a new vector representing the resulting sum if the operation is valid, or -1 if the vectors have incompatible dimensions. Two vectors (lists) can be summed element-wise only if they are of the same length.

Example:
Input:
a = [1, 3], b = [4, 5]
Output:
[5, 8]
Reasoning:
Element-wise sum: [1+4, 3+5] = [5, 8].

"""
def vector_sum(a: list[int|float], b: list[int|float]) -> list[int|float]:
	# Return the element-wise sum of vectors 'a' and 'b'.
	# If vectors have different lengths, return -1.
	if len(a) != len(b): return -1
	return [a[i] + b[i] for i in range(len(a))]
