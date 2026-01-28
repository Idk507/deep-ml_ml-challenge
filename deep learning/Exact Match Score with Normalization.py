"""
Implement a function that calculates the Exact Match (EM) score between a list of predicted strings and a list of reference (ground truth) strings.

The Exact Match score is a common evaluation metric in NLP tasks like question answering and text generation. It measures the proportion of predictions that exactly match the corresponding references.

To make the comparison more robust, your function should normalize both predictions and references before comparison. The normalization should:

Convert text to lowercase
Remove all punctuation characters
Collapse multiple whitespace characters into single spaces and strip leading/trailing whitespace
The function should return a float representing the proportion of exact matches (between 0.0 and 1.0).

If both input lists are empty, return 0.0.

Example:
Input:
predictions = ['Hello, World!', 'The answer is 42'], references = ['hello world', 'the answer is 42']
Output:
1.0
Reasoning:
After normalization, 'Hello, World!' becomes 'hello world' (lowercase, punctuation removed) which matches the reference 'hello world'. Similarly, 'The answer is 42' becomes 'the answer is 42' which matches. Both predictions match their references, so EM = 2/2 = 1.0

Learn About topic
Exact Match Score
The Exact Match (EM) score is a strict evaluation metric commonly used in NLP tasks such as:

Question Answering (e.g., SQuAD benchmark)
Text Generation
Named Entity Recognition
Machine Reading Comprehension
Definition
The Exact Match score measures the percentage of predictions that exactly match the ground truth references:

E
M
=
1
N
∑
i
=
1
N
1
[
pred
i
=
ref
i
]
EM= 
N
1
​
  
i=1
∑
N
​
 1[pred 
i
​
 =ref 
i
​
 ]
Where:

N
N is the total number of examples
1
[
⋅
]
1[⋅] is the indicator function (1 if true, 0 if false)
pred
i
pred 
i
​
  is the 
i
i-th prediction
ref
i
ref 
i
​
  is the 
i
i-th reference
Why Normalization?
Raw string comparison is often too strict. Consider these cases:

Case differences: "Paris" vs "paris"
Punctuation: "Hello!" vs "Hello"
Whitespace: "New York" vs "New York"
Without normalization, these would all be considered non-matches despite being semantically equivalent.

Common Normalization Steps
Lowercase conversion: Ensures case-insensitive matching
Punctuation removal: Ignores punctuation marks like periods, commas, etc.
Whitespace normalization: Collapses multiple spaces and trims edges
Properties
Range: 
E
M
∈
[
0
,
1
]
EM∈[0,1]
Interpretation: Higher is better; 1.0 means perfect predictions
Strictness: Even one character difference results in no match
Comparison with Other Metrics
EM is binary (match or no match)
F1 Score gives partial credit for token overlap
BLEU/ROUGE consider n-gram overlap
Example Calculation
Given:

Predictions: ["The cat", "A dog", "Hello!"]
References: ["the cat", "a cat", "hello"]
After normalization:

"the cat" == "the cat" (match)
"a dog" != "a cat" (no match)
"hello" == "hello" (match)
E
M
=
2
3
≈
0.667
EM= 
3
2
​
 ≈0.667

 """

import string

def exact_match_score(predictions: list[str], references: list[str]) -> float:
    """
    Calculate the exact match score between predictions and references.
    
    Args:
        predictions: List of predicted strings
        references: List of reference (ground truth) strings
    
    Returns:
        Exact match score as a float between 0 and 1
    """
    # Handle empty lists
    if len(predictions) == 0 or len(references) == 0:
        return 0.0
    
    def normalize(text: str) -> str:
        """
        Normalize text by:
        1. Converting to lowercase
        2. Removing punctuation
        3. Normalizing whitespace
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation
        # Create translation table that maps each punctuation char to None
        translator = str.maketrans('', '', string.punctuation)
        text = text.translate(translator)
        
        # Normalize whitespace: split and rejoin to collapse multiple spaces
        # and strip leading/trailing whitespace
        text = ' '.join(text.split())
        
        return text
    
    # Count exact matches
    matches = 0
    n = len(predictions)
    
    for pred, ref in zip(predictions, references):
        # Normalize both strings
        norm_pred = normalize(pred)
        norm_ref = normalize(ref)
        
        # Check for exact match
        if norm_pred == norm_ref:
            matches += 1
    
    # Calculate and return the proportion
    return matches / n
