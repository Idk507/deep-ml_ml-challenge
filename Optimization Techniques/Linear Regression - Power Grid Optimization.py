"""
It is the year 2157. Mars has its first thriving colony, and energy consumption is steadily on the rise. As the lead data scientist, you have daily power usage measurements (10 days) affected by both a growing linear trend and a daily fluctuation. The fluctuation follows the formula fáµ¢ = 10 x sin(2Ï x i / 10), where i is the day number (1 through 10). Your challenge is to remove this known fluctuation from each data point, fit a linear regression model to the detrended data, predict day 15's base consumption, add back the fluctuation for day 15, and finally include a 5% safety margin. The final answer must be an integer, ensuring you meet the colony's future needs.

Example:
Input:
Daily consumption data for 10 days: [150, 165, 185, 195, 210, 225, 240, 260, 275, 290]
Output:
404
Reasoning:
For each of the 10 days, we subtract the daily fluctuation given by 10xsin(2πxi/10). We then perform linear regression on the resulting values, predict day 15’s base usage, and add back the day 15 fluctuation. Finally, we apply a 5% margin. Running the provided solution code yields 404 for this dataset.

Balancing Trend and Fluctuation with Math
When dealing with time-series data, it's common to see both a long-term trend and periodic fluctuations. In this challenge, the daily fluctuation for day 
i
i is given by:

f
i
=
10
×
sin
⁡
(
2
π
×
i
10
)
.
f 
i
​
 =10×sin( 
10
2π×i
​
 ).
Steps to Solve
Fluctuation Removal: Subtract 
f
i
f 
i
​
  from each day's consumption to isolate the colony's base usage.

Linear Regression: Fit a linear model 
y
=
m
x
+
b
y=mx+b using the detrended values. The slope 
m
m and intercept 
b
b are calculated using the least squares method:

m
=
n
∑
(
x
i
y
i
)
−
(
∑
x
i
)
(
∑
y
i
)
n
∑
(
x
i
2
)
−
(
∑
x
i
)
2
,
b
=
∑
y
i
−
m
∑
x
i
n
.
m= 
n∑(x 
i
2
​
 )−(∑x 
i
​
 ) 
2
 
n∑(x 
i
​
 y 
i
​
 )−(∑x 
i
​
 )(∑y 
i
​
 )
​
 ,b= 
n
∑y 
i
​
 −m∑x 
i
​
 
​
 .
Here, 
n
n is the number of data points (10 in this case).

Forecast: Use the regression line to predict the base consumption for day 15, 
x
=
15
x=15:
base
15
=
m
×
15
+
b
.
base 
15
​
 =m×15+b.
Add Back Fluctuation: Compute 
f
15
=
10
×
sin
⁡
(
2
π
×
15
10
)
f 
15
​
 =10×sin( 
10
2π×15
​
 ) and add it to the base prediction:
pred
15
=
base
15
+
f
15
.
pred 
15
​
 =base 
15
​
 +f 
15
​
 .
Round and Add Safety Margin: Round 
pred
15
pred 
15
​
  to the nearest integer and then apply a 5% upward safety margin to ensure sufficient capacity:
final
15
=
⌈
1.05
×
round
(
pred
15
)
⌉
.
final 
15
​
 =⌈1.05×round(pred 
15
​
 )⌉.
Summary
By following these steps removing the fluctuation, fitting the linear model, predicting day 15's base consumption, restoring the fluctuation, and applying a safety margin you'll arrive at a robust energy requirement forecast for the colony's future needs.

"""
import math

PI = 3.14159

def power_grid_forecast(consumption_data):
    n = len(consumption_data)
    x_vals = list(range(1, n + 1))
    
    # Step 1: Remove fluctuation
    detrended = []
    for i in range(n):
        fluctuation = 10 * math.sin(2 * PI * (i + 1) / 10)
        detrended.append(consumption_data[i] - fluctuation)
    
    # Step 2: Linear regression (least squares)
    sum_x = sum(x_vals)
    sum_y = sum(detrended)
    sum_xy = sum(x_vals[i] * detrended[i] for i in range(n))
    sum_x2 = sum(x ** 2 for x in x_vals)
    
    m = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
    b = (sum_y - m * sum_x) / n
    
    # Step 3: Predict base consumption for day 15
    base_15 = m * 15 + b
    
    # Step 4: Add back fluctuation for day 15
    fluctuation_15 = 10 * math.sin(2 * PI * 15 / 10)
    pred_15 = base_15 + fluctuation_15
    
    # Step 5: Round and apply 5% safety margin
    rounded = round(pred_15)
    final = math.ceil(1.05 * rounded)
    
    return final
