import pandas as pd

# Paths
rec_path = "../Product_Recommendation_List.csv"
eval_path = "LLM_Automatic_Product_Evaluation.csv"

# Load
rec = pd.read_csv(rec_path)
ev = pd.read_csv(eval_path)

# Normalize stage
rec["stage"] = rec["stage"].str.lower().str.strip()
ev["stage"] = ev["stage"].str.lower().str.strip()

# Merge relevance (optional but useful)
merge_cols = ["query_id", "stage", "rank", "product_id"]
df = rec.merge(
    ev[merge_cols + ["overall_relevant_llm"]],
    on=merge_cols,
    how="left"
)

# Stage A and Stage B Top-1
top1_a = df[(df["stage"] == "stage_a") & (df["rank"] == 1)][
    ["query_id", "product_id", "product_name"]
].rename(columns={
    "product_id": "product_a",
    "product_name": "product_name_a"
})

top1_b = df[(df["stage"] == "stage_b") & (df["rank"] == 1)][
    ["query_id", "product_id", "product_name"]
].rename(columns={
    "product_id": "product_b",
    "product_name": "product_name_b"
})

# Merge A vs B
comparison = top1_a.merge(top1_b, on="query_id", how="inner")

# Did Top-1 change?
comparison["top1_changed"] = comparison["product_a"] != comparison["product_b"]

# Count changes
total_queries = len(comparison)
changed_count = comparison["top1_changed"].sum()
change_rate = round(changed_count / total_queries * 100, 1)

print("Total Queries:", total_queries)
print("Top-1 Changed:", changed_count)
print("Change Rate (%):", change_rate)

# Now check where Stage B Top-1 ranked in Stage A
stagea_ranks = df[df["stage"] == "stage_a"][
    ["query_id", "product_id", "rank"]
]

origin = top1_b.merge(
    stagea_ranks,
    left_on=["query_id", "product_b"],
    right_on=["query_id", "product_id"],
    how="left"
)

origin = origin[["query_id", "product_b", "rank"]].rename(columns={
    "rank": "rank_in_stage_a"
})

print("\nWhere Stage B Top-1 ranked in Stage A:")
print(origin["rank_in_stage_a"].value_counts(dropna=False))
