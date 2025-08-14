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
    if not transactions:
        raise ValueError('Transaction list cannot be empty')
    if not 0 < min_support <= 1:
        raise ValueError('Minimum support must be between 0 and 1')

    num_transactions = len(transactions)
    min_support_count = min_support * num_transactions
    item_counts = defaultdict(int)
    for transaction in transactions:
        for item in transaction:
            item_counts[frozenset([item])] += 1
    frequent_itemsets = {itemset: count for itemset, count in item_counts.items() if count >= min_support_count}
    k = 1
    all_frequent_itemsets = dict(frequent_itemsets)
    while frequent_itemsets and (max_length is None or k < max_length):
        k += 1
        candidates = generate_candidates(frequent_itemsets.keys(), k)
        candidate_counts = defaultdict(int)
        for transaction in transactions:
            transaction_set = frozenset(transaction)
            for candidate in candidates:
                if candidate.issubset(transaction_set):
                    candidate_counts[candidate] += 1
        frequent_itemsets = {itemset: count for itemset, count in candidate_counts.items() if count >= min_support_count}
        all_frequent_itemsets.update(frequent_itemsets)
    return {itemset: count / num_transactions for itemset, count in all_frequent_itemsets.items()}

def generate_candidates(prev_frequent_itemsets, k):
    candidates = set()
    prev_frequent_list = sorted(list(prev_frequent_itemsets), key=lambda x: sorted(x))
    for i in range(len(prev_frequent_list)):
        for j in range(i + 1, len(prev_frequent_list)):
            itemset1 = prev_frequent_list[i]
            itemset2 = prev_frequent_list[j]
            if k > 2:
                if sorted(itemset1)[:-1] != sorted(itemset2)[:-1]:
                    continue
            new_candidate = itemset1 | itemset2
            if len(new_candidate) == k:
                candidates.add(new_candidate)
    return candidates
