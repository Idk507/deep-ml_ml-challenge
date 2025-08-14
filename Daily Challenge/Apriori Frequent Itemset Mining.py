"""
Implement the Apriori algorithm to discover all frequent itemsets in a set of transactions, given a minimum support threshold. Your function should return all frequent itemsets (of any size) whose support is at least the given minimum. Return the frequent itemsets as a dictionary mapping frozenset of items to their support (fractional). Only use built-in Python and standard libraries (collections, itertools).

Example:
Input:
transactions = [
    {'bread', 'milk'},
    {'bread', 'diaper', 'beer', 'eggs'},
    {'milk', 'diaper', 'beer', 'cola'},
    {'bread', 'milk', 'diaper', 'beer'},
    {'bread', 'milk', 'diaper', 'cola'}
]
result = apriori(transactions, min_support=0.6)
for k in sorted(result, key=lambda x: (len(x), sorted(x))):
    print(sorted(list(k)), round(result[k], 2))
Output:
['bread'] 0.8
['diaper'] 0.8
['milk'] 0.8
['bread', 'diaper'] 0.6
['bread', 'milk'] 0.6
['milk', 'diaper'] 0.6
Reasoning:
Bread, Milk, and Diaper each appear in 4 out of 5 transactions (support=0.8), and the 2-itemsets {'bread','milk'}, {'bread','diaper'}, and {'milk','diaper'} each appear in 3 out of 5 (support=0.6). Only these satisfy the min_support threshold.
Apriori Algorithm for Frequent Itemset Mining
The Apriori algorithm is a classic approach to discovering frequent itemsets for association rule mining in transactional data.

How Apriori Works
Goal:

Find all itemsets that appear in at least a given fraction (the minimum support) of transactions.
Algorithm steps:

Start by counting support for all single items (1-itemsets).
Remove any that do not meet the min_support threshold.
Iteratively:
Use frequent k-itemsets to generate candidate (k+1)-itemsets.
Count supports for these candidates.
Remove those not meeting min_support.
Repeat until no new frequent itemsets are found, or you reach max_length (if given).
Key Principle:

Anti-monotonicity: If an itemset is infrequent, all supersets are also infrequent. This allows Apriori to prune candidates efficiently.
Example:

For the dataset:
T1: {Bread, Milk}
T2: {Bread, Diaper, Beer}
T3: {Milk, Diaper, Beer}
T4: {Bread, Milk, Diaper}
T5: {Bread, Milk, Cola}
With min_support = 0.6 (must appear in â¥3 of 5 transactions):
1-itemsets: Bread (4/5), Milk (4/5), Diaper (3/5) are frequent
2-itemsets: {Bread, Milk} (3/5) is frequent
No charts required: this task is about implementation and set logic.

Applications:

Market basket analysis
Discovering associations in medical, web, or transactional data
"""
import itertools
from collections import defaultdict

def apriori(transactions, min_support=0.5, max_length=None):
    total_transactions = len(transactions)
    min_count = min_support * total_transactions
    transactions = list(map(set, transactions))  # Ensure all transactions are sets

    # Step 1: Count 1-itemsets
    item_counts = defaultdict(int)
    for transaction in transactions:
        for item in transaction:
            item_counts[frozenset([item])] += 1

    # Filter by min_support
    frequent_itemsets = {
        itemset: count / total_transactions
        for itemset, count in item_counts.items()
        if count >= min_count
    }

    current_frequents = set(frequent_itemsets.keys())
    k = 2

    while current_frequents:
        # Stop if max_length is reached
        if max_length and k > max_length:
            break

        # Generate candidate k-itemsets from frequent (k-1)-itemsets
        candidates = set()
        items = set()
        for itemset in current_frequents:
            items |= itemset
        for combo in itertools.combinations(items, k):
            candidate = frozenset(combo)
            # Prune: all (k-1)-subsets must be frequent
            if all(frozenset(subset) in current_frequents for subset in itertools.combinations(candidate, k - 1)):
                candidates.add(candidate)

        # Count support for candidates
        candidate_counts = defaultdict(int)
        for transaction in transactions:
            for candidate in candidates:
                if candidate.issubset(transaction):
                    candidate_counts[candidate] += 1

        # Filter by min_support
        current_frequents = {
            itemset: count / total_transactions
            for itemset, count in candidate_counts.items()
            if count >= min_count
        }

        # Add to global frequent itemsets
        frequent_itemsets.update(current_frequents)
        current_frequents = set(current_frequents.keys())
        k += 1

    return frequent_itemsets
