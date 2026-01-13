def birthday_problem(n: int, days: int = 365) -> float:
    """
    Calculate the probability that at least two people share the same birthday.
    
    Args:
        n (int): Number of people in the group
        days (int): Number of days in a year (default 365)
    
    Returns:
        float: Probability of at least one shared birthday, rounded to 4 decimal places
    """
    # Edge cases
    if n <= 1:
        return 0.0
    if n > days:  # Pigeonhole principle
        return 1.0

    # Probability that no two people share a birthday
    prob_no_match = 1.0
    for i in range(n):
        prob_no_match *= (days - i) / days

    # Probability of at least one match
    prob_match = 1 - prob_no_match
    return round(prob_match, 4)
