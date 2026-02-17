import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load evaluation results
df = pd.read_csv("Eval_Results_Per_Query.csv")

# List of metrics
metrics = [
    "precision_at_5",
    "mrr_at_5",
    "ndcg_at_5",
    "hit_rate_at_5"
]

# Create histograms
for metric in metrics:
    plt.figure(figsize=(8, 5))
    sns.histplot(df[metric], bins=10, kde=True)
    plt.title(f"Distribution of {metric} Across 40 Queries")
    plt.xlabel(metric)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(f"histogram_{metric}.png", dpi=300)
    plt.show()
