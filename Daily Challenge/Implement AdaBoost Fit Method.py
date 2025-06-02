"""
Write a Python function adaboost_fit that implements the fit method for an AdaBoost classifier. The function should take in a 2D numpy array X of shape (n_samples, n_features) representing the dataset, a 1D numpy array y of shape (n_samples,) representing the labels, and an integer n_clf representing the number of classifiers. The function should initialize sample weights, find the best thresholds for each feature, calculate the error, update weights, and return a list of classifiers with their parameters.

Example:
Input:
X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
    y = np.array([1, 1, -1, -1])
    n_clf = 3

    clfs = adaboost_fit(X, y, n_clf)
    print(clfs)
Output:
(example format, actual values may vary):
    # [{'polarity': 1, 'threshold': 2, 'feature_index': 0, 'alpha': 0.5},
    #  {'polarity': -1, 'threshold': 3, 'feature_index': 1, 'alpha': 0.3},
    #  {'polarity': 1, 'threshold': 4, 'feature_index': 0, 'alpha': 0.2}]
Reasoning:
The function fits an AdaBoost classifier on the dataset X with the given labels y and number of classifiers n_clf. It returns a list of classifiers with their parameters, including the polarity, threshold, feature index, and alpha values

"""
import math
import numpy as np
def adaboost_fit(X, y, n_clf):
    n_samples, n_features = np.shape(X)
    w = np.full(n_samples, (1 / n_samples))
    clfs = []
    
    for _ in range(n_clf):
        clf = {}
        min_error = float('inf')
        
        for feature_i in range(n_features):
            feature_values = np.expand_dims(X[:, feature_i], axis=1)
            unique_values = np.unique(feature_values)
            
            for threshold in unique_values:
                p = 1
                prediction = np.ones(np.shape(y))
                prediction[X[:, feature_i] < threshold] = -1
                error = sum(w[y != prediction])
                
                if error > 0.5:
                    error = 1 - error
                    p = -1
                
                if error < min_error:
                    clf['polarity'] = p
                    clf['threshold'] = threshold
                    clf['feature_index'] = feature_i
                    min_error = error
        
        clf['alpha'] = 0.5 * math.log((1.0 - min_error) / (min_error + 1e-10))
        predictions = np.ones(np.shape(y))
        negative_idx = (X[:, clf['feature_index']] < clf['threshold'])
        if clf['polarity'] == -1:
            negative_idx = np.logical_not(negative_idx)
        predictions[negative_idx] = -1
        w *= np.exp(-clf['alpha'] * y * predictions)
        w /= np.sum(w)
        clfs.append(clf)

    return clfs
