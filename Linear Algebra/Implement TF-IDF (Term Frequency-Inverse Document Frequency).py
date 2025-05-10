"""
Implement TF-IDF (Term Frequency-Inverse Document Frequency)

Task: Implement TF-IDF (Term Frequency-Inverse Document Frequency)
Your task is to implement a function that computes the TF-IDF scores for a query against a given corpus of documents.

Function Signature
Write a function compute_tf_idf(corpus, query) that takes the following inputs:

corpus: A list of documents, where each document is a list of words.
query: A list of words for which you want to compute the TF-IDF scores.
Output
The function should return a list of lists containing the TF-IDF scores for the query words in each document, rounded to five decimal places.

Important Considerations
Handling Division by Zero:
When implementing the Inverse Document Frequency (IDF) calculation, you must account for cases where a term does not appear in any document (df = 0). This can lead to division by zero in the standard IDF formula. Add smoothing (e.g., adding 1 to both numerator and denominator) to avoid such errors.

Empty Corpus:
Ensure your implementation gracefully handles the case of an empty corpus. If no documents are provided, your function should either raise an appropriate error or return an empty result. This will ensure the program remains robust and predictable.

Edge Cases:

Query terms not present in the corpus.
Documents with no words.
Extremely large or small values for term frequencies or document frequencies.
By addressing these considerations, your implementation will be robust and handle real-world scenarios effectively.

Example:
Input:
corpus = [
    ["the", "cat", "sat", "on", "the", "mat"],
    ["the", "dog", "chased", "the", "cat"],
    ["the", "bird", "flew", "over", "the", "mat"]
]
query = ["cat"]

print(compute_tf_idf(corpus, query))
Output:
[[0.21461], [0.25754], [0.0]]
Reasoning:
The TF-IDF scores for the word "cat" in each document are computed and rounded to five decimal places.

Understanding TF-IDF (Term Frequency-Inverse Document Frequency)
TF-IDF is a numerical statistic that reflects how important a word is in a document relative to a collection (or corpus). It is widely used in information retrieval, text mining, and natural language processing tasks.

Mathematical Formulation
TF-IDF is the product of two key statistics: Term Frequency (TF) and Inverse Document Frequency (IDF).

1. Term Frequency (TF)
The term frequency is defined as:

T
F
(
t
,
d
)
=
Number of times term 
t
 appears in document 
d
Total number of terms in document 
d
TF(t,d)= 
Total number of terms in document d
Number of times term t appears in document d
​
 

t
t: A specific term (word).
d
d: A specific document in the corpus.
2. Inverse Document Frequency (IDF)
To account for how common or rare a term is across all documents in the corpus, we calculate:

I
D
F
(
t
)
=
log
⁡
(
N
+
1
df
(
t
)
+
1
)
+
1
IDF(t)=log( 
df(t)+1
N+1
​
 )+1

Where:

N
N: Total number of documents in the corpus.
df
(
t
)
df(t): Number of documents containing the term 
t
t.
Adding 
+
1
+1 inside the fraction prevents division by zero if a term never appears.
Adding 
+
1
+1 outside the log ensures IDF remains nonzero.
3. TF-IDF
Combining TF and IDF:

T
F
I
D
F
(
t
,
d
)
=
T
F
(
t
,
d
)
×
I
D
F
(
t
)
TFIDF(t,d)=TF(t,d)×IDF(t)

Implementation Steps
Compute TF
For each document, count how often each term appears and divide by the documentâs total word count.

Compute IDF
For each term, calculate its document frequency across all documents and apply the IDF formula.

Calculate TF-IDF
For every term in every document, multiply the termâs TF by its IDF.

Normalization (Optional)
Normalize TF-IDF vectors (e.g., using 
L
2
L2 norm) if comparing documents in a vector space model.

Example Calculation
Suppose we have a small corpus of 3 documents:

Doc1: "The cat sat on the mat"
Doc2: "The dog chased the cat"
Doc3: "The bird flew over the mat"
We want to calculate the TF-IDF for the word "cat" in Doc1.

Step 1: Compute 
T
F
(
"
c
a
t
"
,
Doc1
)
TF("cat",Doc1)
T
F
(
"
c
a
t
"
,
Doc1
)
=
1
6
≈
0.1667
TF("cat",Doc1)= 
6
1
​
 ≈0.1667

"cat" appears once.
Total words in Doc1 (counting each occurrence of âtheâ) = 6.
Step 2: Compute 
I
D
F
(
"
c
a
t
"
)
IDF("cat")
"cat" appears in 2 out of 3 documents, so 
df
(
"
c
a
t
"
)
=
2
df("cat")=2.
N
=
3
N=3.
Using the formula with smoothing and an added constant:

I
D
F
(
"
c
a
t
"
)
=
log
⁡
(
N
+
1
df
(
"
c
a
t
"
)
+
1
)
+
1
=
log
⁡
(
3
+
1
2
+
1
)
+
1
=
log
⁡
(
4
3
)
+
1
≈
0.2877
+
1
=
1.2877
IDF("cat")=log( 
df("cat")+1
N+1
​
 )+1=log( 
2+1
3+1
​
 )+1=log( 
3
4
​
 )+1≈0.2877+1=1.2877

Step 3: Calculate 
T
F
I
D
F
(
"
c
a
t
"
,
Doc1
)
TFIDF("cat",Doc1)
T
F
I
D
F
(
"
c
a
t
"
,
Doc1
)
=
T
F
(
"
c
a
t
"
,
Doc1
)
×
I
D
F
(
"
c
a
t
"
)
=
0.1667
×
1.2877
≈
0.2147
TFIDF("cat",Doc1)=TF("cat",Doc1)×IDF("cat")=0.1667×1.2877≈0.2147

Applications of TF-IDF
Information Retrieval
TF-IDF is often used in search engines to rank how relevant a document is to a given query.
Text Mining
Helps identify key terms and topics in large volumes of text.
Document Classification
Useful for weighting important words in classification tasks.
Search Engines
Refines document ranking by emphasizing distinctive terms.
Recommendation Systems
Evaluates text-based similarity (e.g., for content-based filtering).
TF-IDF remains a foundational technique in natural language processing, widely used for feature extraction and analysis across numerous text-based applications.
"""
import numpy as np

def compute_tf_idf(corpus, query):
    """
    Compute TF-IDF scores for a query against a corpus of documents using only NumPy.
    The output TF-IDF scores retain five decimal places.
    """
    vocab = sorted(set(word for document in corpus for word in document).union(query))
    word_to_index = {word: idx for idx, word in enumerate(vocab)}

    tf = np.zeros((len(corpus), len(vocab)))

    for doc_idx, document in enumerate(corpus):
        for word in document:
            word_idx = word_to_index[word]
            tf[doc_idx, word_idx] += 1
        tf[doc_idx, :] /= len(document)

    df = np.count_nonzero(tf > 0, axis=0)

    num_docs = len(corpus)
    idf = np.log((num_docs + 1) / (df + 1)) + 1

    tf_idf = tf * idf

    query_indices = [word_to_index[word] for word in query]
    tf_idf_scores = tf_idf[:, query_indices]

    tf_idf_scores = np.round(tf_idf_scores, 5)

    return tf_idf_scores.tolist()
