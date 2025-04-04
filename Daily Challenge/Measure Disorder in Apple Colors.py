"""
Measure Disorder in Apple Colors

Implement a function that calculates the disorder in a basket of apples based on their colors, where each apple color is represented by an integer. The disorder must be 0 if all apples are the same color and must increase as the variety of colors increases. In particular:

[0,0,0,0] should yield 0.
[1,1,0,0] should have a higher disorder than [0,0,0,0].
[0,1,2,3] should have a higher disorder than [1,1,0,0].
[0,0,1,1,2,2,3,3] should have a higher disorder than [0,0,0,0,0,1,2,3].
You may use any method to measure disorder as long as these properties are satisfied.

Example:
Input:
disorder([1,1,0,0])
Output:
0.5 #or any value from -inf till +inf
Reasoning:
In the basket [1,1,0,0], there are two distinct colors each appearing with equal frequency (0.5).

Using Gini Impurity to Measure Disorder
One valid approach to measure disorder in a basket of apples is to use the Gini impurity metric. The Gini impurity is defined as:

G
=
1
−
∑
i
=
1
k
p
i
2
G=1− 
i=1
∑
k
​
 p 
i
2
​
 
where:

p
i
p 
i
​
  is the proportion of apples of the 
i
i-th color.
k
k is the total number of distinct colors.
Key Properties
Single Color Case: If all apples in the basket have the same color, then 
p
=
1
p=1 and the Gini impurity is:
G
=
1
−
1
2
=
0
G=1−1 
2
 =0
Increasing Disorder: As the variety of colors increases, the impurity increases. For example:
Two equally frequent colors:
G
=
1
−
(
0.5
2
+
0.5
2
)
=
0.5
G=1−(0.5 
2
 +0.5 
2
 )=0.5
Four equally frequent colors:
G
=
1
−
(
4
×
0.25
2
)
=
0.75
G=1−(4×0.25 
2
 )=0.75
Comparing Different Baskets
Basket: [0,0,0,0]
Only one color -> 
G
=
0
G=0
Basket: [1,1,0,0]
Two colors, equal frequency -> 
G
=
0.5
G=0.5
Basket: [0,1,2,3]
Four equally frequent colors -> 
G
=
0.75
G=0.75
Basket: [0,0,1,1,2,2,3,3]
Equal distribution among four colors -> 
G
=
0.75
G=0.75
Basket: [0,0,0,0,0,1,2,3]
One dominant color, three others -> 
G
=
0.5625
G=0.5625
Flexibility
While the Gini impurity is a suitable measure of disorder, any method that satisfies the following constraints is valid:

A basket with a single color must return a disorder score of 0.
Baskets with more distinct colors must yield higher disorder scores.
The specific ordering constraints provided in the problem must be maintained.
By using this impurity measure, we can quantify how diverse a basket of apples is based on color distribution.


"""

def disorder(apples: list) -> float:
    """
    Calculates a measure of disorder in a basket of apples based on their colors.
    One valid approach is to use the Gini impurity, defined as:
      G = 1 - sum((count/total)^2 for each color)
    This method returns 0 for a basket with all apples of the same color and increases as the variety of colors increases.
    While this implementation uses the Gini impurity, any method that satisfies the following properties is acceptable:
      1. A single color results in a disorder of 0.
      2. Baskets with more distinct colors yield a higher disorder score.
      3. The ordering constraints are maintained.
    """
    if not apples:
        return 0.0
    total = len(apples)
    counts = {}
    for color in apples:
        counts[color] = counts.get(color, 0) + 1
    impurity = 1.0
    for count in counts.values():
        p = count / total
        impurity -= p * p
    return round(impurity, 4)
