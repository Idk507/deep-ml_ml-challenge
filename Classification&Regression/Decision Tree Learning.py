
"""
Write a Python function that implements the decision tree learning algorithm for classification. The function should use recursive binary splitting based on entropy and information gain to build a decision tree. It should take a list of examples (each example is a dict of attribute-value pairs) and a list of attribute names as input, and return a nested dictionary representing the decision tree.

Example:
Input:
examples = [
                    {'Outlook': 'Sunny', 'Temperature': 'Hot', 'Humidity': 'High', 'Wind': 'Weak', 'PlayTennis': 'No'},
                    {'Outlook': 'Sunny', 'Temperature': 'Hot', 'Humidity': 'High', 'Wind': 'Strong', 'PlayTennis': 'No'},
                    {'Outlook': 'Overcast', 'Temperature': 'Hot', 'Humidity': 'High', 'Wind': 'Weak', 'PlayTennis': 'Yes'},
                    {'Outlook': 'Rain', 'Temperature': 'Mild', 'Humidity': 'High', 'Wind': 'Weak', 'PlayTennis': 'Yes'}
                ],
                attributes = ['Outlook', 'Temperature', 'Humidity', 'Wind']
Output:
{
            'Outlook': {
                'Sunny': {'Humidity': {'High': 'No', 'Normal': 'Yes'}},
                'Overcast': 'Yes',
                'Rain': {'Wind': {'Weak': 'Yes', 'Strong': 'No'}}
            }
        }
Reasoning:
Using the given examples, the decision tree algorithm determines that 'Outlook' is the best attribute to split the data initially. When 'Outlook' is 'Overcast', the outcome is always 'Yes', so it becomes a leaf node. In cases of 'Sunny' and 'Rain', it further splits based on 'Humidity' and 'Wind', respectively. The resulting tree structure is able to classify the training examples with the attributes 'Outlook', 'Temperature', 'Humidity', and 'Wind'.

Learn About topic
Decision Tree Learning Algorithm
The decision tree learning algorithm is a method used for classification that predicts the value of a target variable based on several input variables. Each internal node of the tree corresponds to an input variable, and each leaf node corresponds to a class label.

Algorithm Overview
The recursive binary splitting starts by selecting the attribute that best separates the examples according to the entropy and information gain, calculated as follows:

Entropy
H
(
X
)
=
−
∑
p
(
x
)
log
⁡
2
p
(
x
)
H(X)=−∑p(x)log 
2
​
 p(x)
Information Gain
I
G
(
D
,
A
)
=
H
(
D
)
−
∑
∣
D
v
∣
∣
D
∣
H
(
D
v
)
IG(D,A)=H(D)−∑ 
∣D∣
∣D 
v
​
 ∣
​
 H(D 
v
​
 )
Explanation of Terms
Entropy ( H(X) ): Measures the impurity or disorder of the set.
Information Gain ( IG(D, A) ): Represents the reduction in entropy after splitting the dataset ( D ) on attribute ( A ).
( D_v ): The subset of ( D ) for which attribute ( A ) has value ( v ).
Process
Select Attribute: Choose the attribute with the highest information gain.
Split Dataset: Divide the dataset based on the values of the selected attribute.
Recursion: Repeat the process for each subset until:
All data is perfectly classified, or
No remaining attributes can be used to make a split.
This recursive process continues until the decision tree can no longer be split further or all examples have been classified.
"""




import math
from collections import Counter

def calculate_entropy(labels):
    label_counts = Counter(labels)
    total_count = len(labels)
    entropy = -sum((count / total_count) * math.log2(count / total_count) for count in label_counts.values())
    return entropy

def calculate_information_gain(examples, attr, target_attr):
    total_entropy = calculate_entropy([example[target_attr] for example in examples])
    values = set(example[attr] for example in examples)
    attr_entropy = 0
    for value in values:
        value_subset = [example[target_attr] for example in examples if example[attr] == value]
        value_entropy = calculate_entropy(value_subset)
        attr_entropy += (len(value_subset) / len(examples)) * value_entropy
    return total_entropy - attr_entropy

def majority_class(examples, target_attr):
    return Counter([example[target_attr] for example in examples]).most_common(1)[0][0]

def learn_decision_tree(examples, attributes, target_attr):
    if not examples:
        return 'No examples'
    if all(example[target_attr] == examples[0][target_attr] for example in examples):
        return examples[0][target_attr]
    if not attributes:
        return majority_class(examples, target_attr)
    
    gains = {attr: calculate_information_gain(examples, attr, target_attr) for attr in attributes}
    best_attr = max(gains, key=gains.get)
    tree = {best_attr: {}}
    
    for value in set(example[best_attr] for example in examples):
        subset = [example for example in examples if example[best_attr] == value]
        new_attributes = attributes.copy()
        new_attributes.remove(best_attr)
        subtree = learn_decision_tree(subset, new_attributes, target_attr)
        tree[best_attr][value] = subtree
    
    return tree
