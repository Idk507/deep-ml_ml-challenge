"""
Write a Python function to perform Group Normalization on a 4D input tensor with shape (B, C, H, W). The function should normalize over smaller groups of channels, then apply a learned scale (gamma) and shift (beta).

Example:
Input:
X.shape = (2, 2, 2, 2), gamma = [1, 1], beta = [0, 0], num_groups = 2
Output:
Normalized tensor where each group is independently normalized and scaled by gamma and shifted by beta.
Reasoning:
First split the channels into groups, compute mean and variance per group, normalize within the group, then scale and shift with gamma and beta.

"""
import numpy as np
def group_normalization(X: np.ndarray, gamma: np.ndarray, beta: np.ndarray, num_groups: int, epsilon: float = 1e-5) -> np.ndarray:
    
    batchsize,numchannel,height,width = X.shape 
    groupsize = numchannel //  num_groups
    Xreshape = X.reshape(batchsize,num_groups,groupsize,height,width)
    mean = np.mean(Xreshape,axis=(2,3,4),keepdims = True)
    variance = np.var(Xreshape,axis=(2,3,4),keepdims = True)

    Xnorm = (Xreshape -  mean) / np.sqrt(variance + epsilon)
    Xnorm = Xnorm.reshape(batchsize,numchannel,height,width)
    normx = gamma* Xnorm + beta 
    return normx
