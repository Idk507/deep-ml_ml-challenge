"""
Create a function to decide when to stop training a model early based on a list of validation losses. The early stopping criterion should stop training if the validation loss hasn't improved for a specified number of epochs (patience), and only count as improvement if the loss decreases by more than a certain threshold (min_delta). Your function should return the epoch to stop at and the best epoch that achieved the lowest validation loss.

Example:
Input:
[0.9, 0.8, 0.75, 0.77, 0.76, 0.77, 0.78], patience=2, min_delta=0.01
Output:
(4, 2)
Reasoning:
The best validation loss is 0.75 at epoch 2. There is no improvement greater than 0.01 for the next 2 epochs. Therefore, training should stop at epoch 4.
Implementing Early Stopping Criterion
Early stopping is a regularization technique that helps prevent overfitting in machine learning models. Your task is to implement the early stopping decision logic based on the validation loss history.

Problem Description
Given a sequence of validation losses from model training, determine if training should be stopped based on the following criteria:

Training should stop if the validation loss hasn't improved (decreased) for a specified number of epochs (patience)
An improvement is only counted if the loss decreases by more than a minimum threshold (min_delta)
The best model is the one with the lowest validation loss
Example
Consider the following validation losses: [0.9, 0.8, 0.75, 0.77, 0.76, 0.77, 0.78]

With patience=2 and min_delta=0.01:
Best loss is 0.75 at epoch 2
No improvement > 0.01 for next 2 epochs
Should stop at epoch 4
Function Requirements
Return both the epoch to stop at and the best epoch
If no stopping is needed, return the last epoch
Epochs are 0-indexed
"""
from typing import Tuple

def early_stopping(val_losses: list[float], patience: int, min_delta: float) -> Tuple[int, int]:
    # Your code here
    best_loss = float('inf')
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch, loss in enumerate(val_losses):
        if loss < best_loss - min_delta:
            best_loss = loss
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            return epoch, best_epoch

    return len(val_losses) - 1, best_epoch
