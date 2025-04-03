"""
Calculate R-squared for Regression Analysis

Task: Compute the R-squared Value in Regression Analysis
R-squared, also known as the coefficient of determination, is a measure that indicates how well the independent variables explain the variability of the dependent variable in a regression model.

Your Task: To implement the function r_squared(y_true, y_pred) that calculates the R-squared value, given arrays of true values y_true and predicted values y_pred.

Example:
Input:
import numpy as np

y_true = np.array([1, 2, 3, 4, 5])
y_pred = np.array([1.1, 2.1, 2.9, 4.2, 4.8])
print(r_squared(y_true, y_pred))
Output:
0.989
Reasoning:
The R-squared value is calculated to be 0.989, indicating that the regression model explains 98.9% of the variance in the dependent variable.

Learn About topic
Understanding R-squared (RÂ²) in Regression Analysis
R-squared, also known as the coefficient of determination, is a statistical measure that represents the proportion of the variance for a dependent variable that's explained by an independent variable or variables in a regression model. It provides insight into how well the model fits the data.

Mathematical Definition
The R-squared value is calculated using the following formula:

R
2
=
1
−
SSR
SST
R 
2
 =1− 
SST
SSR
​
 
Where:

SSR
SSR (Sum of Squared Residuals): The sum of the squares of the differences between the actual values and the predicted values.
SST
SST (Total Sum of Squares): The sum of the squares of the differences between the actual values and the mean of the actual values.
Equations for SSR and SST
To calculate SSR and SST, we use the following formulas:

SSR:
SSR
=
∑
i
=
1
n
(
y
i
−
y
^
i
)
2
SSR= 
i=1
∑
n
​
 (y 
i
​
 − 
y
^
​
  
i
​
 ) 
2
 
SST:
SST
=
∑
i
=
1
n
(
y
i
−
y
ˉ
)
2
SST= 
i=1
∑
n
​
 (y 
i
​
 − 
y
ˉ
​
 ) 
2
 
Where:

y
i
y 
i
​
 : Actual value
y
^
i
y
^
​
  
i
​
 : Predicted value
y
ˉ
y
ˉ
​
 : Mean of the actual values
Significance of R-squared
R-squared is a key metric for evaluating how well a regression model performs. A higher R-squared value indicates a better fit for the model, meaning it can explain more variability in the data. However, it's important to note:

A high R-squared does not always imply that the model is good; it can sometimes be misleading if overfitting occurs.
It should be used in conjunction with other metrics for comprehensive model evaluation.
Implementing R-squared Calculation
In this problem, you will implement a function to calculate R-squared given arrays of true and predicted values from a regression task. The results should be rounded to three decimal places.

In the solution, the implemented 
r
_
s
q
u
a
r
e
d
(
)
r_squared() function calculates R-squared by first determining SSR and SST, then applying them to compute 
R
2
R 
2
"""
import numpy as np

def r_squared(y_true, y_pred):
    """
    Calculate the R-squared (RÂ²) coefficient of determination.
    
    Args:
        y_true (numpy.ndarray): Array of true values
        y_pred (numpy.ndarray): Array of predicted values
    
    Returns:
        float: R-squared value rounded to 3 decimal places
    """
    if np.array_equal(y_true, y_pred):
        return 1.0

    # Calculate mean of true values
    y_mean = np.mean(y_true)

    # Calculate Sum of Squared Residuals (SSR)
    ssr = np.sum((y_true - y_pred) ** 2)

    # Calculate Total Sum of Squares (SST)
    sst = np.sum((y_true - y_mean) ** 2)

    try:
        # Calculate R-squared
        r2 = 1 - (ssr / sst)
        if np.isinf(r2):
            return 0.0
        return round(r2, 3)
    except ZeroDivisionError:
        return 0.0
edge cases such as perfect predictions and situations where all true values are identical.

Reference
You can refer to this resource for more information:
Coefficient of Determination

""'
