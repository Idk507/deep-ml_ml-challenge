"""
Write a Python function convert_range that shifts and scales the values of a NumPy array from their original range 
[
a
,
b
]
[a,b] (where 
a
=
min
⁡
(
x
)
a=min(x) and 
b
=
max
⁡
(
x
)
b=max(x)) to a new target range 
[
c
,
d
]
[c,d]. Your function should work for both 1D and 2D arrays, returning an array of the same shape, and only use NumPy. Return floating-point results, and ensure you use the correct formula to map the input interval to the output interval.

Example:
Input:
import numpy as np
x = np.array([0, 5, 10])
c, d = 2, 4
print(convert_range(x, c, d))
Output:
[2. 3. 4.]
Reasoning:
The minimum value (a) is 0 and the maximum value (b) is 10. The formula maps 0 to 2, 5 to 3, and 10 to 4 using: f(x) = c + (d-c)/(b-a)*(x-a).

Shifting and Scaling a Range (Rescaling Data)
1. Motivation
Rescaling (or shifting and scaling) is a common preprocessing step in data analysis and machine learning. It's often necessary to map data from an original range (e.g., test scores, pixel values, GPA) to a new range suitable for downstream tasks or compatibility between datasets. For example, you might want to shift a GPA from 
[
0
,
10
]
[0,10] to 
[
0
,
4
]
[0,4] for comparison or model input.

2. The General Mapping Formula
Suppose you have input values in the range 
[
a
,
b
]
[a,b] and you want to map them to the interval 
[
c
,
d
]
[c,d].

First, shift the lower bound to 
0
0 by applying 
x
↦
x
−
a
x↦x−a, so 
[
a
,
b
]
→
[
0
,
b
−
a
]
[a,b]→[0,b−a].
Next, scale to unit interval: 
t
↦
1
b
−
a
⋅
t
t↦ 
b−a
1
​
 ⋅t, yielding 
[
0
,
1
]
[0,1].
Now, scale to 
[
0
,
d
−
c
]
[0,d−c] with 
t
↦
(
d
−
c
)
t
t↦(d−c)t, and shift to 
[
c
,
d
]
[c,d] with 
t
↦
c
+
t
t↦c+t.
Combining all steps, the complete formula is:
f
(
x
)
=
c
+
(
d
−
c
b
−
a
)
(
x
−
a
)
f(x)=c+( 
b−a
d−c
​
 )(x−a)
x
x = the input value
a
=
min
⁡
(
x
)
a=min(x) and 
b
=
max
⁡
(
x
)
b=max(x)
c
c, 
d
d = target interval endpoints
3. Applications
Image Processing: Rescale pixel intensities
Feature Engineering: Normalize features to a common range
Score Conversion: Convert test scores or grades between systems
4. Practical Considerations
Be aware of the case when 
a
=
b
a=b (constant input); this may require special handling (e.g., output all 
c
c).
For multidimensional arrays, use NumPyâs .min() and .max() to determine the full input range.
This formula gives a simple, mathematically justified way to shift and scale data to any target rangeâa core tool for robust machine learning pipelines.


"""
import numpy as np

def convert_range(values: np.ndarray, c: float, d: float) -> np.ndarray:
    """
    Shift and scale values from their original range [min, max] to a target [c, d] range.
    """
    # Your code here
    a = min(values)
    b = max(values)
    
    if a == b : return np.full_like(values,fill_value = c,dtype = np.float64)

    scaled = c + ((d - c)/(b - a)) * (values - a)
    return scaled.astype(np.float64)
