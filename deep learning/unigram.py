"""
Implement a function that calculates the unigram probability of a given word in a corpus of sentences. Include start <s> and end </s> tokens in the calculation. The probability should be rounded to 4 decimal places.

Example:
Input:
corpus = "<s> Jack I like </s> <s> Jack I do like </s>", word = "Jack"
Output:
0.1818
Reasoning:
The corpus has 11 total tokens. 'Jack' appears twice. So, probability = 2 / 11

Unigram Probability Calculation
In Natural Language Processing (NLP), a unigram model is the simplest form of a language model.

It assumes each word in a sentence is generated independently.

The probability of a word w under the unigram model is:

P
(
w
)
=
Count
(
w
)
∑
w
′
∈
V
Count
(
w
′
)
P(w)= 
∑ 
w 
′
 ∈V
​
 Count(w 
′
 )
Count(w)
​
 

Where:

Count
(
w
)
Count(w) = Number of times the word w appears in the corpus.

V
V = Vocabulary (all word tokens in the corpus).

∑
w
′
∈
V
Count
(
w
′
)
∑ 
w 
′
 ∈V
​
 Count(w 
′
 ) = Total number of word tokens.

Round upto the 4th decimal point.

Sample Corpus
<s> I am Jack </s>
<s> Jack I am </s>
<s> Jack I like </s>
<s> Jack I do like </s>
<s> do I like Jack </s>
Notes :

<s> : Start of a sentence
</s> : End of a sentence
Need to count both the start and enod of sentence tokens while calculating probability.
Zero probability issues are not addressed here and will be covered separately under smoothing techniques in later problems.

"""
def unigram_probability(corpus: str, word: str) -> float:
    # Your code here
    tokens = corpus.split()
    word_count = len(tokens)
    word_per_token = tokens.count(word)
    return round(word_per_token/word_count,4)
