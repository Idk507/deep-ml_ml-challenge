"""
Captain Redbeard, the most daring pirate of the seven seas, has uncovered a mysterious ancient map. Instead of islands, it shows a strange wavy curve, and the treasure lies at the lowest point of the land! (watch out for those tricky local mins)

The land's height at any point 
x
x is given by:

f(x) = x^4 - 3x^3 + 2

Your Mission: Implement a Python function that finds the value of 
x
x where 
f
(
x
)
f(x) reaches its minimum, starting from any random initial position.

Example:
Input:
start_x = 0.0
Output:
min float value

How to Find the Minimum of a Function
To find the minimum of a function like

f
(
x
)
=
x
4
−
3
x
3
+
2
f(x)=x 
4
 −3x 
3
 +2
we can use a technique called gradient descent.

Steps:
Find the Derivative

The derivative (slope) tells us which direction the function is increasing or decreasing.
For this problem, the derivative is:
f
′
(
x
)
=
4
x
3
−
9
x
2
f 
′
 (x)=4x 
3
 −9x 
2
 
Move Opposite the Slope

If the slope is positive, move left.
If the slope is negative, move right.
Update the position by:
x
n
e
w
=
x
o
l
d
−
learning rate
×
f
′
(
x
o
l
d
)
x 
new
​
 =x 
old
​
 −learning rate×f 
′
 (x 
old
​
 )
Repeat

Keep updating 
x
x until the change is very small (below a tolerance).
Why Does This Work?
If you always move downhill along the slope, you eventually reach a bottom a local minimum.
Important Terms
Learning Rate: How big a step to take each update.
Tolerance: How close successive steps must be to stop.
Local Minimum: A point where the function value is lower than nearby points.
In this problem, Captain Redbeard finds the hidden treasure by moving downhill until he reaches the lowest point!
"""
def find_treasure(start_x: float) -> float:
	learning_rate = 0.1, tolerance = 1e-6, max_iters = 10000
  def gradient(x):
        return 4 * x**3 - 9 * x**2  # derivative of x^4 - 3x^3 + 2

    x = start_x
    for _ in range(max_iters):
        grad = gradient(x)
        new_x = x - learning_rate * grad
        if abs(new_x - x) < tolerance:
            break
        x = new_x
    return round(x, 4)
