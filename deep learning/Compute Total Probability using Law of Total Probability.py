def law_of_total_probability(priors: dict, conditionals: dict) -> float:
    """
    Compute P(A) using the Law of Total Probability.
    
    Args:
        priors: Dictionary mapping partition event names to P(Bi)
        conditionals: Dictionary mapping partition event names to P(A|Bi)
    
    Returns:
        float: The total probability P(A), rounded to 4 decimal places
    """
    # Compute the sum of P(A|Bi) * P(Bi) for all partition events
    total_probability = sum(
        conditionals[event] * priors[event] 
        for event in priors
    )
    
    # Round to 4 decimal places
    return round(total_probability, 4)
