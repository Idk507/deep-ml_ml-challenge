"""
Implement INT8 Quantization
Medium
Deep Learning

Implement a function that performs symmetric INT8 quantization on floating-point values. INT8 quantization is a technique used in deep learning to reduce model size and speed up inference by converting 32-bit floating-point numbers to 8-bit integers.

Your function should:

Calculate the appropriate scale factor for symmetric quantization
Quantize the input values to signed 8-bit integers in the range [-127, 127]
Dequantize the values back to floating-point to demonstrate the reconstruction
For symmetric quantization:

The scale factor maps the maximum absolute value to 127
The zero point is always 0 (symmetric around origin)
Quantized values are clipped to the valid INT8 range
Return a dictionary containing:

'quantized': List of quantized integer values
'scale': The scale factor used (rounded to 6 decimal places)
'dequantized': List of reconstructed floating-point values (rounded to 4 decimal places)
Handle the edge case where all input values are zero by using a scale of 1.0.

Example:
Input:
x = [1.0, -1.0, 0.5, -0.5]
Output:
{'quantized': [127, -127, 64, -64], 'scale': 0.007874, 'dequantized': [1.0, -1.0, 0.5039, -0.5039]}
Reasoning:
Find abs_max = max(|1.0|, |-1.0|, |0.5|, |-0.5|) = 1.0
Calculate scale = 1.0 / 127 = 0.007874
Quantize each value: 1.0/0.007874 = 127, -1.0/0.007874 = -127, 0.5/0.007874 = 63.5 -> round to 64, -0.5/0.007874 = -63.5 -> round to -64
Dequantize: 127 * 0.007874 = 1.0, -127 * 0.007874 = -1.0, 64 * 0.007874 = 0.5039, -64 * 0.007874 = -0.5039 Note the quantization error for 0.5 and -0.5 due to the discrete nature of INT8 representation.
Learn About topic
INT8 Quantization for Deep Learning
What is Quantization?
Quantization is the process of mapping continuous values (like 32-bit floating-point numbers) to a discrete set of values (like 8-bit integers). In deep learning, quantization reduces model size and increases inference speed, often with minimal loss in accuracy.

Why INT8?
INT8 quantization is popular because:

4x memory reduction: INT8 uses 8 bits vs 32 bits for FP32
Faster computation: Integer operations are faster than floating-point
Hardware support: Many accelerators have optimized INT8 instructions
Symmetric Quantization
Symmetric quantization maps floating-point values symmetrically around zero. The key formulas are:

Scale Factor: 
scale
=
max
⁡
(
∣
x
∣
)
127
scale= 
127
max(∣x∣)
​
 

We use 127 instead of 128 to keep the range symmetric: 
[
−
127
,
127
]
[−127,127].

Quantization: 
q
=
clip
(
round
(
x
scale
)
,
−
127
,
127
)
q=clip(round( 
scale
x
​
 ),−127,127)

Dequantization: 
x
^
=
q
×
scale
x
^
 =q×scale

Quantization Error
The quantization process introduces error because we're mapping continuous values to discrete integers. The maximum quantization error for symmetric INT8 is:

ϵ
m
a
x
=
scale
2
ϵ 
max
​
 = 
2
scale
​
 

This error is why 0.5 becomes 0.5039 in our example - the value 63.5 rounds to 64, which maps back to a slightly different value.

Example Walkthrough
For input 
x
=
[
1.0
,
−
1.0
,
0.5
,
−
0.5
]
x=[1.0,−1.0,0.5,−0.5]:

Find maximum absolute value: 
∣
x
∣
m
a
x
=
1.0
∣x∣ 
max
​
 =1.0

Calculate scale: 
scale
=
1.0
127
≈
0.007874
scale= 
127
1.0
​
 ≈0.007874

Quantize each value:

q
1
=
round
(
1.0
/
0.007874
)
=
round
(
127
)
=
127
q 
1
​
 =round(1.0/0.007874)=round(127)=127
q
2
=
round
(
−
1.0
/
0.007874
)
=
round
(
−
127
)
=
−
127
q 
2
​
 =round(−1.0/0.007874)=round(−127)=−127
q
3
=
round
(
0.5
/
0.007874
)
=
round
(
63.5
)
=
64
q 
3
​
 =round(0.5/0.007874)=round(63.5)=64
q
4
=
round
(
−
0.5
/
0.007874
)
=
round
(
−
63.5
)
=
−
64
q 
4
​
 =round(−0.5/0.007874)=round(−63.5)=−64
Dequantize:

x
^
1
=
127
×
0.007874
=
1.0
x
^
  
1
​
 =127×0.007874=1.0
x
^
2
=
−
127
×
0.007874
=
−
1.0
x
^
  
2
​
 =−127×0.007874=−1.0
x
^
3
=
64
×
0.007874
=
0.5039
x
^
  
3
​
 =64×0.007874=0.5039
x
^
4
=
−
64
×
0.007874
=
−
0.5039
x
^
  
4
​
 =−64×0.007874=−0.5039
Asymmetric vs Symmetric Quantization
Symmetric quantization (zero point = 0) is simpler but may waste dynamic range for asymmetric distributions. Asymmetric quantization uses a non-zero offset:

q
=
round
(
x
scale
)
+
zero_point
q=round( 
scale
x
​
 )+zero_point

Symmetric quantization is preferred for weights (usually centered around zero), while asymmetric may be better for activations (often non-negative after ReLU).
"""

import numpy as np

def int8_quantize(x: list[float]) -> dict:
  
    Perform symmetric INT8 quantization on a floating-point array.

    Args:
        x: Input list of floating-point values

    Returns:
        Dictionary with 'quantized', 'scale', and 'dequantized' keys
    
    if not x:  # handle empty input
        return {'quantized': [], 'scale': 1.0, 'dequantized': []}

    # Step 1: Find maximum absolute value
    abs_max = max(abs(val) for val in x)

    # Step 2: Handle edge case (all zeros)
    scale = 1.0 if abs_max == 0 else abs_max / 127.0

    # Step 3: Quantize values
    quantized = []
    for val in x:
        q = round(val / scale) if scale != 0 else 0
        q = max(-127, min(127, q))  # clip to [-127, 127]
        quantized.append(q)

    # Step 4: Dequantize values
    dequantized = [round(q * scale, 4) for q in quantized]

    return {
        'quantized': quantized,
        'scale': round(scale, 6),
        'dequantized': dequantized
    }
