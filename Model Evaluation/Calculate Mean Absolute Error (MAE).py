"""
Calculate Mean Absolute Error (MAE)

Implement a function to calculate the Mean Absolute Error (MAE) between two arrays of actual and predicted values. The MAE is a metric used to measure the average magnitude of errors in a set of predictions without considering their direction.

Example:
Input:
y_true = np.array([3, -0.5, 2, 7]), y_pred = np.array([2.5, 0.0, 2, 8])
Output:
0.500
Reasoning:
The MAE is calculated by taking the mean of the absolute differences between the predicted and true values. Using the formula, the result is 0.500.

"""
import numpy as np

def mae(y_true, y_pred):
	"""
	Calculate Mean Absolute Error between two arrays.

	Parameters:
	y_true (numpy.ndarray): Array of true values
    y_pred (numpy.ndarray): Array of predicted values

	Returns:
	float: Mean Absolute Error rounded to 3 decimal places
	"""
	return round(np.mean(np.abs(y_true - y_pred)), 3)
