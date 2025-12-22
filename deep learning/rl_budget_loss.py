import numpy as np

def rl_budget_loss(
    rewards: np.ndarray,
    log_probs: np.ndarray,
    old_log_probs: np.ndarray,
    response_lengths: np.ndarray,
    token_budget: int,
    kl_coef: float,
    budget_penalty_coef: float
) -> float:
    """
    Compute the budget-constrained RL loss used in Kimi K2.

    Loss:
        E[(A_i - KL_i)^2]

    where:
        A_i  = adjusted_reward_i - baseline
        KL_i = kl_coef * (log_probs_i - old_log_probs_i)

    Args:
        rewards: (batch_size, K)
        log_probs: (batch_size, K)
        old_log_probs: (batch_size, K)
        response_lengths: (batch_size, K)
        token_budget: int
        kl_coef: τ
        budget_penalty_coef: λ

    Returns:
        Scalar loss
    """

    # 1. Compute budget penalty: λ * max(0, L_i - B)
    excess_tokens = np.maximum(0, response_lengths - token_budget)
    budget_penalty = budget_penalty_coef * excess_tokens

    # 2. Adjust rewards
    adjusted_rewards = rewards - budget_penalty

    # 3. Compute baseline (mean per prompt)
    baseline = adjusted_rewards.mean(axis=1, keepdims=True)

    # 4. Compute advantages
    advantages = adjusted_rewards - baseline

    # 5. KL regularization term
    kl_term = kl_coef * (log_probs - old_log_probs)

    # 6. Squared loss
    loss = np.mean((advantages - kl_term) ** 2)

    return float(loss)
