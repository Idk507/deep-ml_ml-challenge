"""
Implement a Domain Expert Fusion algorithm inspired by LongCat-Flash-Thinking's domain-parallel training scheme.

In this approach, instead of training a single model on all domains simultaneously (which can be unstable), separate expert models are trained for different domains (e.g., STEM, Code, Agentic). These experts are then fused into a single model that achieves near Pareto-optimal performance across all domains.

Given:

expert_scores: A dictionary mapping expert names to their performance scores across domains. Each expert has scores for multiple domains as a dictionary {domain: score}.
domain_weights: A dictionary mapping domain names to their importance weights (weights sum to 1.0).
fusion_method: Either 'weighted_average' or 'best_per_domain'
Your function should:

For 'weighted_average': Compute a weighted combination of all experts, where the fused model's score for each domain is the average of all experts' scores for that domain, weighted by how well each expert performs on that domain relative to others.
For 'best_per_domain': For each domain, select the score from the expert that performs best on that domain.
Finally, compute the overall fused score as the weighted sum of domain scores using domain_weights.

Return the overall fused score as a float.

Example:
Input:
expert_scores = {
    'stem_expert': {'math': 95, 'code': 70, 'agentic': 60},
    'code_expert': {'math': 65, 'code': 92, 'agentic': 75},
    'agent_expert': {'math': 55, 'code': 68, 'agentic': 94}
}
domain_weights = {'math': 0.4, 'code': 0.35, 'agentic': 0.25}
fusion_method = 'best_per_domain'
Output:
93.7
Reasoning:
Using best_per_domain: math=95 (from stem_expert), code=92 (from code_expert), agentic=94 (from agent_expert). Overall = 950.4 + 920.35 + 94*0.25 = 38 + 32.2 + 23.5 = 93.7. The fused model achieves near Pareto-optimal performance by taking the best of each specialist.

Learn About topic
Domain-Parallel Expert Fusion
The Problem with Mixed-Domain Training
Training a single model on multiple domains (STEM, Code, Agentic) simultaneously can lead to:

Optimization instability: Conflicting gradients across domains
Catastrophic forgetting: Improving one domain degrades another
Suboptimal tradeoffs: No single model excels everywhere
LongCat-Flash-Thinking's Solution
The domain-parallel training scheme:

Train Separate Experts: Optimize independent models for each domain
Measure Performance: Evaluate each expert across all domains
Fuse into One Model: Combine experts into a unified model
Fusion Strategies
Best-Per-Domain: 
fused
d
=
max
⁡
e
(
score
e
,
d
)
fused 
d
​
 =max 
e
​
 (score 
e,d
​
 )

Takes the best expert's score for each domain. Achieves theoretical upper bound.

Weighted Average: 
fused
d
=
∑
e
w
e
,
d
⋅
score
e
,
d
∑
e
w
e
,
d
fused 
d
​
 = 
∑ 
e
​
 w 
e,d
​
 
∑ 
e
​
 w 
e,d
​
 ⋅score 
e,d
​
 
​
 

Weights experts by their relative strength on each domain.

Pareto Optimality
A solution is Pareto-optimal if no domain can be improved without degrading another. Expert fusion aims to find a point on the Pareto frontier where the fused model takes the best characteristics from each specialist.

Key Results from LongCat
Domain-parallel training stabilizes optimization
Fused model achieves state-of-the-art across benchmarks
Enables specialization without sacrificing generalization
"""
def domain_expert_fusion(expert_scores: dict, domain_weights: dict, fusion_method: str = 'weighted_average') -> float:
    """
    Fuse domain expert models into a single unified model.
    """
    domains = list(domain_weights.keys())
    fused_domain_scores = {}
    
    if fusion_method == 'best_per_domain':
        # Take best expert per domain
        for domain in domains:
            fused_domain_scores[domain] = max(
                scores[domain] for scores in expert_scores.values()
            )
    
    elif fusion_method == 'weighted_average':
        # Weighted average based on relative performance
        for domain in domains:
            # Collect all expert scores for this domain
            scores = [expert_scores[exp][domain] for exp in expert_scores]
            total = sum(scores)
            
            if total == 0:
                fused_domain_scores[domain] = 0.0
            else:
                # Each expert weighted by (score / total_score)
                # This gives higher weight to better-performing experts
                fused_domain_scores[domain] = sum(
                    score * (score / total) for score in scores
                )
    
    else:
        raise ValueError(f"Unknown fusion method: {fusion_method}")
    
    # Compute weighted sum across domains
    overall_score = sum(
        fused_domain_scores[domain] * domain_weights[domain]
        for domain in domains
    )
    
    return float(overall_score)
