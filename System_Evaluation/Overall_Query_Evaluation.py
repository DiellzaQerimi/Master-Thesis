
import pandas as pd

# Load per-query evaluation results
per_query_path = "Eval_Results_Per_Query.csv"
df = pd.read_csv(per_query_path)

# Compute overall statistics
overall_stats = pd.DataFrame({
    "Metric": ["Precision@5", "Hit@5", "MRR@5", "NDCG@5"],
    "Mean": [
        df["precision_at_k"].mean(),
        df["hit_at_k"].mean(),
        df["mrr_at_k"].mean(),
        df["ndcg_at_k"].mean()
    ],
    "Standard Deviation": [
        df["precision_at_k"].std(),
        df["hit_at_k"].std(),
        df["mrr_at_k"].std(),
        df["ndcg_at_k"].std()
    ],
    "Minimum": [
        df["precision_at_k"].min(),
        df["hit_at_k"].min(),
        df["mrr_at_k"].min(),
        df["ndcg_at_k"].min()
    ],
    "Maximum": [
        df["precision_at_k"].max(),
        df["hit_at_k"].max(),
        df["mrr_at_k"].max(),
        df["ndcg_at_k"].max()
    ]
})

# Round for thesis presentation
overall_stats = overall_stats.round(3)

print(overall_stats)

