import math

def thanksgiving_feast_predictor(preference_scores):
    """
    Convert preference scores into a probability distribution using softmax.
    
    Args:
        preference_scores (list of float): Scores for dishes (e.g., turkey, stuffing, cranberry, pie).
    
    Returns:
        list of float: Probabilities summing to 1.
    """
    # Numerical stability trick: subtract max score
    max_score = max(preference_scores)
    exp_scores = [math.exp(score - max_score) for score in preference_scores]
    
    total = sum(exp_scores)
    probabilities = [round(exp_score / total, 4) for exp_score in exp_scores]
    
    return probabilities
