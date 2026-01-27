"""
Implement Out-of-Bag Score Calculation
Medium
Machine Learning

In bagging ensemble methods like Random Forest, each base estimator is trained on a bootstrap sample (random sample with replacement) of the training data. This means that for each estimator, some samples from the original dataset are not used during training - these are called Out-of-Bag (OOB) samples.

The OOB score provides an unbiased estimate of the ensemble's generalization performance without requiring a separate validation set. For each sample in the dataset, we can aggregate predictions from all estimators for which that sample was OOB, then compare the aggregated prediction with the true label.

Your task is to implement a function calculate_oob_score that computes the OOB accuracy score for a classification task.

The function takes:

n_samples: Total number of samples in the original dataset
bootstrap_indices: A list of lists, where each inner list contains the indices of samples used to train each estimator (samples NOT in this list are OOB for that estimator)
predictions: A list of lists, where each inner list contains that estimator's predictions for ALL samples in the dataset
y_true: The true labels for all samples
The function should:

Identify which samples are OOB for each estimator
Collect OOB predictions for each sample
Aggregate predictions using majority voting
Return the accuracy score over all samples that have at least one OOB prediction
If a sample has no OOB predictions (it was in-bag for all estimators), exclude it from the calculation. If no samples have OOB predictions, return 0.0.

Example:
Input:
n_samples=5, bootstrap_indices=[[0, 1, 2], [1, 2, 3], [2, 3, 4]], predictions=[[0, 1, 1, 1, 0], [0, 1, 1, 1, 0], [0, 1, 1, 1, 0]], y_true=[0, 1, 1, 1, 0]
Output:
1.0
Reasoning:
For each sample, we identify which estimators did NOT use it: Sample 0 is OOB for estimators 1 and 2 (not in their bootstrap indices), so we collect their predictions [0, 0] and majority vote gives 0, which matches y_true[0]=0. Sample 1 is OOB for estimator 2 only, prediction is 1, matches y_true[1]=1. Sample 2 is in-bag for all estimators, so it's excluded. Sample 3 is OOB for estimator 0, prediction is 1, matches y_true[3]=1. Sample 4 is OOB for estimators 0 and 1, predictions are [0, 0], majority vote is 0, matches y_true[4]=0. All 4 OOB samples are correctly predicted, so OOB score = 4/4 = 1.0

"""
import numpy as np

def calculate_oob_score(n_samples: int, bootstrap_indices: list, predictions: list, y_true: list) -> float:
    """
    Calculate the Out-of-Bag score for a bagging ensemble.
    
    Args:
        n_samples: Total number of samples in the dataset
        bootstrap_indices: List of lists containing indices used to train each estimator
        predictions: List of lists containing predictions from each estimator for all samples
        y_true: True labels for all samples
    
    Returns:
        OOB accuracy score as a float
    """
    # Convert y_true to numpy array for easier indexing
    y_true = np.array(y_true)
    
    # Track OOB predictions for each sample
    oob_predictions = [[] for _ in range(n_samples)]
    
    # For each estimator
    n_estimators = len(bootstrap_indices)
    for estimator_idx in range(n_estimators):
        # Get the bootstrap indices (in-bag samples) for this estimator
        in_bag_indices = set(bootstrap_indices[estimator_idx])
        
        # Get predictions from this estimator
        estimator_preds = predictions[estimator_idx]
        
        # For each sample, check if it's OOB for this estimator
        for sample_idx in range(n_samples):
            if sample_idx not in in_bag_indices:
                # This sample is OOB for this estimator
                oob_predictions[sample_idx].append(estimator_preds[sample_idx])
    
    # Calculate OOB score
    correct = 0
    total_oob_samples = 0
    
    for sample_idx in range(n_samples):
        # Check if this sample has any OOB predictions
        if len(oob_predictions[sample_idx]) > 0:
            # Use majority voting to get the final prediction
            oob_preds = oob_predictions[sample_idx]
            
            # Find the most common prediction (majority vote)
            unique_preds, counts = np.unique(oob_preds, return_counts=True)
            majority_vote = unique_preds[np.argmax(counts)]
            
            # Check if the majority vote matches the true label
            if majority_vote == y_true[sample_idx]:
                correct += 1
            
            total_oob_samples += 1
    
    # Return accuracy
    if total_oob_samples == 0:
        return 0.0
    
    return correct / total_oob_samples
