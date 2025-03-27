"""
Descriptive Statistics Calculator

Write a Python function to calculate various descriptive statistics metrics for a given dataset. The function should take a list or NumPy array of numerical values and return a dictionary containing mean, median, mode, variance, standard deviation, percentiles (25th, 50th, 75th), and interquartile range (IQR).

Example:
Input:
[10, 20, 30, 40, 50]
Output:
{'mean': 30.0, 'median': 30.0, 'mode': 10, 'variance': 200.0, 'standard_deviation': 14.142135623730951, '25th_percentile': 20.0, '50th_percentile': 30.0, '75th_percentile': 40.0, 'interquartile_range': 20.0}
Reasoning:
The dataset is processed to calculate all descriptive statistics. The mean is the average value, the median is the central value, the mode is the most frequent value, and variance and standard deviation measure the spread of data. Percentiles and IQR describe data distribution.
"""
import numpy as np 
def descriptive_statistics(data):
        mean = np.mean(data)
        median = np.median(data)
        unique, counts = np.unique(data, return_counts=True)
        mode = unique[np.argmax(counts)] if len(data) > 0 else None
        variance = np.var(data)
        std_dev = np.sqrt(variance)
        percentiles = np.percentile(data, [25, 50, 75])
        iqr = percentiles[2] - percentiles[0]
        stats_dict = {
            "mean": mean,
            "median": median,
            "mode": mode,
            "variance": np.round(variance,4),
            "standard_deviation": np.round(std_dev,4),
            "25th_percentile": percentiles[0],
            "50th_percentile": percentiles[1],
            "75th_percentile": percentiles[2],
            "interquartile_range": iqr
        }
        return stat_dict
