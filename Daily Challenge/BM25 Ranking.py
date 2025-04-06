"""
BM25 Ranking

Implement the BM25 ranking function to calculate document scores for a query in an information retrieval context. BM25 is an advanced variation of TF-IDF that incorporates term frequency saturation, document length normalization, and a configurable penalty for document length effects.

Example:
Input:
corpus = [['the', 'cat', 'sat'], ['the', 'dog', 'ran'], ['the', 'bird', 'flew']], query = ['the', 'cat']
Output:
[0.693, 0., 0. ]
Reasoning:
BM25 calculates scores for each document in the corpus by evaluating how well the query terms match each document while considering term frequency saturation and document length normalization.

"""
"""
BM25
BM25 (Best Match 25) is used in information retrieval for search relevance. Similar to TF-IDF, it reflects the importance of a word in a document within a collection or corpus. However, BM25 improves upon TF-IDF by addressing key limitations.

Limitations of TF-IDF Addressed by BM25
Saturation: In TF-IDF, having a term multiple times in a document skews the term frequency, making the document overly relevant. BM25 mitigates this by using:
TF-adjusted
=
TF
TF
+
k
1
TF-adjusted= 
TF+k 
1
​
 
TF
​
 

Document Length Normalization: BM25 accounts for document length by normalizing term frequencies using:
Normalized Length
=
1
−
b
+
b
×
Doc Len
Average Doc Len
Normalized Length=1−b+b× 
Average Doc Len
Doc Len
​
 

Amplifying Parameter: The 
b
b parameter controls the influence of document length normalization. Higher 
b
b values amplify the effect.

Final BM25 Formula
The BM25 score for a term is given by:
BM25
=
IDF
×
TF
×
(
k
1
+
1
)
TF
+
k
1
×
(
1
−
b
+
b
×
dl
adl
)
BM25=IDF× 
TF+k 
1
​
 ×(1−b+b× 
adl
dl
​
 )
TF×(k 
1
​
 +1)
​
 

Where:

TF
TF: Term frequency in the document.
IDF
IDF: Inverse document frequency, calculated as 
log
⁡
(
N
+
1
df
+
1
)
log( 
df+1
N+1
​
 ).
N
N: Total number of documents.
df
df: Number of documents containing the term.
dl
dl: Document length.
adl
adl: Average document length.
k
1
k 
1
​
 : Saturation parameter.
b
b: Normalization parameter.
Implementation Steps
Compute document length (
d
l
dl) and average document length (
a
d
l
adl).
Calculate term frequencies (
T
F
TF) using the BM25 formula.
Compute inverse document frequencies (
I
D
F
IDF) for each term.
Calculate BM25 scores for each document.
Applications
BM25 is widely used in:

Search Engines
Recommendation Systems
Natural Language Processing (NLP)
Understanding BM25 enables the creation of robust systems for search and ranking tasks.

"""

import numpy as np
from collections import Counter

def calculate_bm25_scores(corpus, query, k1=1.5, b=0.75):
    if not corpus or not query:
        raise ValueError("Corpus and query cannot be empty")

    doc_lengths = [len(doc) for doc in corpus]
    avg_doc_length = np.mean(doc_lengths)
    doc_term_counts = [Counter(doc) for doc in corpus]
    doc_freqs = Counter()
    for doc in corpus:
        doc_freqs.update(set(doc))

    scores = np.zeros(len(corpus))
    N = len(corpus)

    for term in query:
        df = doc_freqs.get(term, 0) + 1
        idf = np.log((N + 1) / df)

        for idx, term_counts in enumerate(doc_term_counts):
            if term not in term_counts:
                continue

            tf = term_counts[term]
            doc_len_norm = 1 - b + b * (doc_lengths[idx] / avg_doc_length)
            term_score = (tf * (k1 + 1)) / (tf + k1 * doc_len_norm)
            scores[idx] += idf * term_score

    return np.round(scores, 3)
