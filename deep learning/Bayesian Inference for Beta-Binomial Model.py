"""
Implement Bayesian inference for a binomial likelihood with a Beta prior (Beta-Binomial conjugate model). Given prior parameters (alpha, beta) and observed data (successes, trials), compute the posterior distribution parameters and posterior mean. This demonstrates how prior beliefs are updated with observed data using Bayes' theorem.

Example:
Input:
prior_alpha=1, prior_beta=1, successes=7, trials=10
Output:
(8.0, 4.0, 0.6667)
Reasoning:
Prior: Beta(1,1) uniform. Data: 7 successes, 3 failures. Posterior: Beta(1+7, 1+3) = Beta(8, 4). Posterior mean = 8/(8+4) = 0.6667. The uniform prior is updated by the data to give posterior centered at observed proportion.
Bayesian Inference
Bayesian inference updates prior beliefs with observed data to obtain posterior beliefs using Bayes' theorem.

Bayes' Theorem
P
(
θ
∣
data
)
=
P
(
data
∣
θ
)
⋅
P
(
θ
)
P
(
data
)
P(θ∣data)= 
P(data)
P(data∣θ)⋅P(θ)
​
 
In words:

Posterior
=
Likelihood
×
Prior
Evidence
Posterior= 
Evidence
Likelihood×Prior
​
 
Components:

Prior 
P
(
θ
)
P(θ): Belief about 
θ
θ before seeing data
Likelihood 
P
(
data
∣
θ
)
P(data∣θ): Probability of data given 
θ
θ
Posterior 
P
(
θ
∣
data
)
P(θ∣data): Updated belief after seeing data
Evidence 
P
(
data
)
P(data): Normalizing constant
Beta-Binomial Model
This is a conjugate model: prior and posterior have the same family (Beta distribution).

Setup:


"""

import numpy as np

def bayesian_inference_beta_binomial(prior_alpha: float, prior_beta: float, 
                                     successes: int, trials: int) -> tuple[float, float, float]:
    """
    Perform Bayesian inference for Beta-Binomial model.
    
    Args:
        prior_alpha: Alpha parameter of Beta prior
        prior_beta: Beta parameter of Beta prior
        successes: Number of successes observed
        trials: Total number of trials
    
    Returns:
        Tuple of (posterior_alpha, posterior_beta, posterior_mean) where:
        - posterior_alpha: Updated alpha parameter
        - posterior_beta: Updated beta parameter
        - posterior_mean: Mean of posterior distribution
    """
    
    failures = trials - successes
    
    posterior_alpha = prior_alpha + successes
    posterior_beta = prior_beta + failures
    posterior_mean = posterior_alpha / (posterior_alpha + posterior_beta)
    
    return posterior_alpha, posterior_beta, posterior_mean

