import pandas as pd

# Load file
df = pd.read_csv("Eval_Results_Per_Query.csv")

K = 5

# Group by query_type (each should have 10 queries)
summary_by_type = (
    df.groupby("query_type")
      .agg({
          "precision_at_5": "mean",
          "hit_rate_at_5": "mean",
          "mrr_at_5": "mean",
          "ndcg_at_5": "mean",
          "query_id": "count"
      })
      .reset_index()
)

# Rename columns to match thesis table
summary_by_type = summary_by_type.rename(columns={
    "query_id": "Number of Queries",
    "precision_at_5": "Mean Precision@K",
    "hit_rate_at_5": "Mean Hit@Rate",
    "mrr_at_5": "Mean MRR@K",
    "ndcg_at_5": "Mean NDCG@K"
})

# Insert K column at beginning
summary_by_type.insert(0, "K", K)

# Round nicely
summary_by_type = summary_by_type.round(2)

print(summary_by_type)
