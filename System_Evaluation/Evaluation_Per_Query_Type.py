import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load evaluation results
df = pd.read_csv("eval_results_per_query.csv")

# List of metrics
metrics = [
    "precision_at_5",
    "mrr_at_5",
    "ndcg_at_5",
    "hit_rate_at_5"
]

# Create boxplots grouped by query_type
for metric in metrics:
    plt.figure(figsize=(8, 5))
    sns.boxplot(x="query_type", y=metric, data=df)
    plt.title(f"{metric} by Query Type (10 Queries Each)")
    plt.xlabel("Query Type")
    plt.ylabel(metric)
    plt.tight_layout()
    plt.savefig(f"boxplot_{metric}_by_query_type.png", dpi=300)
    plt.show()
