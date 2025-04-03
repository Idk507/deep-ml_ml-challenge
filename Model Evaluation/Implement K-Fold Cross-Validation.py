"""
Implement K-Fold Cross-Validation

Implement a function to generate train and test splits for K-Fold Cross-Validation. Your task is to divide the dataset into k folds and return a list of train-test indices for each fold.

Example:
Input:
k_fold_cross_validation(np.array([0,1,2,3,4,5,6,7,8,9]), np.array([0,1,2,3,4,5,6,7,8,9]), k=5, shuffle=False)
Output:
[([2, 3, 4, 5, 6, 7, 8, 9], [0, 1]), ([0, 1, 4, 5, 6, 7, 8, 9], [2, 3]), ([0, 1, 2, 3, 6, 7, 8, 9], [4, 5]), ([0, 1, 2, 3, 4, 5, 8, 9], [6, 7]), ([0, 1, 2, 3, 4, 5, 6, 7], [8, 9])]
Reasoning:
The function splits the dataset into 5 folds without shuffling and returns train-test splits for each iteration.

Learn About topic
Understanding K-Fold Cross-Validation
K-Fold Cross-Validation is a resampling technique used to evaluate machine learning models by partitioning the dataset into multiple folds.

How it Works
The dataset is split into k equal (or almost equal) parts called folds.
Each fold is used once as a test set, while the remaining k-1 folds form the training set.
The process is repeated k times, ensuring each fold serves as a test set exactly once.
Why Use K-Fold Cross-Validation?
It provides a more robust estimate of model performance than a single train-test split.
Reduces bias introduced by a single training/testing split.
Allows evaluation across multiple data distributions.
Implementation Steps
Shuffle the data if required.
Split the dataset into k equal (or nearly equal) folds.
Iterate over each fold, using it as the test set while using the remaining data as the training set.
Return train-test indices for each iteration.
By implementing this function, you will learn how to split a dataset for cross-validation, a crucial step in model evaluation.


"""
import numpy as np

def k_fold_cross_validation(X: np.ndarray, y: np.ndarray, k=5, shuffle=True, random_seed=None):
    """
    Return train and test indices for k-fold cross-validation.
    """
    n_samples = len(X)
    indices = np.arange(n_samples)
    
    if shuffle:
        if random_seed is not None:
            np.random.seed(random_seed)
        np.random.shuffle(indices)
    
    fold_sizes = np.full(k, n_samples // k, dtype=int)
    fold_sizes[:n_samples % k] += 1

    current = 0
    folds = []
    for fold_size in fold_sizes:
        folds.append(indices[current:current + fold_size])
        current += fold_size

    result = []
    for i in range(k):
        test_idx = folds[i]
        train_idx = np.concatenate(folds[:i] + folds[i+1:])
        result.append((train_idx.tolist(), test_idx.tolist()))
    
    return result
