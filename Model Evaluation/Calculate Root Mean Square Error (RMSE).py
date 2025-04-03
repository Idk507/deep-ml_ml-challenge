"""
Calculate Root Mean Square Error (RMSE)

Task: Compute Root Mean Square Error (RMSE)
In this task, you are required to implement a function rmse(y_true, y_pred) that calculates the Root Mean Square Error (RMSE) between the actual values and the predicted values. RMSE is a commonly used metric for evaluating the accuracy of regression models, providing insight into the standard deviation of residuals.

Your Task:
Implement the function rmse(y_true, y_pred) to:

Calculate the RMSE between the arrays y_true and y_pred.
Return the RMSE value rounded to three decimal places.
Ensure the function handles edge cases such as:
Mismatched array shapes.
Empty arrays.
Invalid input types.
The RMSE is defined as:

RMSE
=
1
n
∑
i
=
1
n
(
y
true
,
i
−
y
pred
,
i
)
2
RMSE= 
n
1
​
  
i=1
∑
n
​
 (y 
true,i
​
 −y 
pred,i
​
 ) 
2
 
​
 
Where:

n
n is the number of observations.
y
true
,
i
y 
true,i
​
  and 
y
pred
,
i
y 
pred,i
​
  are the actual and predicted values for the 
i
i-th observation.
Example:
Input:
y_true = np.array([3, -0.5, 2, 7])
y_pred = np.array([2.5, 0.0, 2, 8])
print(rmse(y_true, y_pred))
Output:
0.612
Reasoning:
The RMSE is calculated as sqrt((0.5^2 + 0.5^2 + 0^2 + 1^2) / 4) = 0.612
"""

return round(np.sqrt(np.mean((y_true - y_pred) ** 2)), 3)
