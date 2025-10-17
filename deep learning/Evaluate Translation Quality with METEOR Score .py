"""
Evaluate Translation Quality with METEOR Score 
Develop a function to compute the METEOR score for evaluating machine translation quality. Given a reference translation and a candidate translation, calculate the score based on unigram matches, precision, recall, F-mean, and a penalty for word order fragmentation.
Example: Input: meteor_score('Rain falls gently from the sky', 'Gentle rain drops from the sky') Output: 0.625 Reasoning: The function identifies 4 unigram matches ('rain', 'gently'/'gentle', 'from', 'sky'), computes precision (4/6) and recall (4/5), calculates an F-mean, and then apply a small penalty for two chunks. METEOR(Metric for Evaluation of Translation with Explicit ORdering) is a metric generally used for machine translation and evaluating the text output of generative AI models. METEOR build was introduced to address the limitations in earlier metrics like BLEU.
Key Characteristics Considers semantic similarity beyond exact word matching Accounts for word order and translation variations Provides more human-aligned translation assessment Implementation Tokenization
Frequency of matching words : Matching needs to be exact
Calculate Precision, Recall and F-mean
F_mean = (Precision * Recall) / (alpha * Precision + (1 - alpha) * Recall) alpha typically set to 0.9 Balances precision and recall Fragmentation Penalty
Chunks = Count of contiguous matched word sequences Penalty = gamma * (Chunks / Matches)^Î² beta controls penalty weight (typically 3) gamma limits maximum penalty (typically 0.5) Final METEOR Score
METEOR = F_mean * (1 - Penalty) Ranges from 0 (no match) to 1 (perfect match) Note : The paper that introduced the metric doesn't have the parameters (alpha,Î², and gamma) as tunable parameters, but implementation in other libraries like NLTK offers this flexibility.
Example Reference: "The quick brown fox jumps over the lazy dog" Candidate: "A quick brown fox jumps over a lazy dog"
- Tokenization Reference Tokens: ['the', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog'] Candidate Tokens: ['a', 'quick', 'brown', 'fox', 'jumps', 'over', 'a', 'lazy', 'dog']
- Unigram Matching Matching tokens: ['quick', 'brown', 'fox', 'jumps', 'over', 'lazy', 'dog'] Matches: 7
- Unigram Precision and Recall Calculation Precision = Matches / Candidate Length = 7 / 9 ~ 0.778
Recall = Matches / Reference Length = 7 / 9 ~ 0.778
- F-mean Calculation (alpha = 0.9) F_mean = (Precision * Recall) / (alpha * Precision + (1 - alpha) * Recall) = (0.778 * 0.778) / (0.9 * 0.778 + (1 - 0.9) * 0.778) = 0.606 / (0.7 + 0.078) = 0.606 / 0.778 â 0.779
- Chunk Calculation Contiguous matched sequences: ['quick', 'brown', 'fox'] ['jumps', 'over'] ['lazy', 'dog'] Number of Chunks: 3 Total Number of Unigram Matches: 7
- Penalty Calculation (betta = 3, gamma = 0.5) Penalty = gamma * (Number of Chunks / Total Number of Unigram Matches)^betta = 0.5 * (3 / 7)^3 = 0.5 * (0.429)^3 â 0.039
- Final METEOR Score METEOR = F_mean * (1 - Penalty) = 0.779 * (1 - 0.039) = 0.779 * 0.961 â 0.749 import numpy as np from collections import Counter

"""
def meteor_score(reference, candidate, alpha=0.9, beta=3, gamma=0.5):
    # Tokenize and lowercase
    ref_tokens = reference.lower().split()
    cand_tokens = candidate.lower().split()

    # Count matches with exact unigram matching
    ref_counts = {}
    for word in ref_tokens:
        ref_counts[word] = ref_counts.get(word, 0) + 1

    matches = []
    used_ref = {}
    for i, word in enumerate(cand_tokens):
        if word in ref_counts and ref_counts[word] > 0:
            matches.append((i, word))
            ref_counts[word] -= 1
            used_ref[word] = used_ref.get(word, []) + [i]

    match_count = len(matches)
    if match_count == 0:
        return 0.0

    # Precision and Recall
    precision = match_count / len(cand_tokens)
    recall = match_count / len(ref_tokens)

    # F-mean
    f_mean = (precision * recall) / (alpha * precision + (1 - alpha) * recall)

    # Chunk calculation
    # Get reference indices for matched words in candidate order
    ref_indices = []
    ref_map = {}
    for i, word in enumerate(ref_tokens):
        ref_map.setdefault(word, []).append(i)

    used_positions = set()
    for _, word in matches:
        for pos in ref_map[word]:
            if pos not in used_positions:
                ref_indices.append(pos)
                used_positions.add(pos)
                break

    # Count chunks: discontinuities in reference indices
    chunks = 1
    for i in range(1, len(ref_indices)):
        if ref_indices[i] != ref_indices[i - 1] + 1:
            chunks += 1

    # Penalty
    penalty = gamma * (chunks / match_count) ** beta

    # Final METEOR score
    meteor = f_mean * (1 - penalty)
    return round(meteor, 3)
