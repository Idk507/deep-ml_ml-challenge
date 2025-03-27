"""
Normal Distribution PDF Calculator

Write a Python function to calculate the probability density function (PDF) of the normal distribution for a given value, mean, and standard deviation. The function should use the mathematical formula of the normal distribution to return the PDF value rounded to 5 decimal places.

Example:
Input:
x = 16, mean = 15, std_dev = 2.04
Output:
0.17342
Reasoning:
The function computes the PDF using x = 16, mean = 15, and std_dev = 2.04.
"""
import math

def normal_pdf(x, mean, std_dev):
        coefficient = 1 / (math.sqrt(2 * math.pi) * std_dev)
        exponent = math.exp(-((x - mean) ** 2) / (2 * std_dev ** 2))
        return round(coefficient * exponent, 5)
