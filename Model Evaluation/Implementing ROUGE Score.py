"""
Implement the ROUGE-1 (Recall-Oriented Understudy for Gisting Evaluation) score to evaluate the quality of a generated summary by comparing it to a reference summary. ROUGE-1 focuses on unigram (single word) overlaps between the candidate and reference texts. Your task is to write a function that computes the ROUGE-1 recall, precision, and F1 score based on the number of overlapping unigrams.

Example:
Input:
rouge_1_score('the cat sat on the mat', 'the cat is on the mat')
Output:
{'precision': 0.8333333333333334, 'recall': 0.8333333333333334, 'f1': 0.8333333333333334}
Reasoning:
The reference text 'the cat sat on the mat' has 6 tokens, and the candidate text 'the cat is on the mat' has 6 tokens. The overlapping words are: 'the' (appears 2 times in reference, 2 times in candidate, so min(2,2)=2 overlap), 'cat' (1,1 → 1 overlap), 'on' (1,1 → 1 overlap), and 'mat' (1,1 → 1 overlap). Total overlap = 2+1+1+1 = 5. Precision = 5/6 ≈ 0.833 (5 overlapping words out of 6 candidate words). Recall = 5/6 ≈ 0.833 (5 overlapping words out of 6 reference words). F1 = 2×(0.833×0.833)/(0.833+0.833) = 0.833 since precision equals recall.
"""
# Implement your function below.
from collections import Counter 

def rouge_1_score(reference: str, candidate: str) -> dict:
    """
    Compute ROUGE-1 score between reference and candidate texts.
    
    Returns a dictionary with precision, recall, and f1.
    """
    # Your code here
    ref_token = reference.lower().split()
    can_token = candidate.lower().split()

    ref_counter = Counter(ref_token)
    can_counter = Counter(can_token)

    overlap = sum((min(ref_counter[w],can_counter[w]) for w in can_counter ))

    precision = overlap / len(can_token) if can_token else  0.0 
    recall = overlap / len(ref_token) if ref_token else 0.0 

    f1 = (2*precision*recall) / (precision + recall) if (preciion + recall) else 0.0 

    return {"precision": precision, "recall": recall, "f1": f1}
