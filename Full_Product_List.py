import pandas as pd

# Load datasets
df_ulta = pd.read_csv("Ulta_Products/Ulta_Product_Details_Cleaned.csv", low_memory=False)
df_sephora = pd.read_csv("Sephora_Products/Sephora_Product_Details_Cleaned.csv", low_memory=False)

# Case-insensitive keys for join
df_ulta["product_lower"] = df_ulta["product"].str.lower()
df_ulta["brand_lower"] = df_ulta["brand"].str.lower()

df_sephora["product_lower"] = df_sephora["product"].str.lower()
df_sephora["brand_lower"] = df_sephora["brand"].str.lower()

# Outer join on product + brand
df_merged = pd.merge(
    df_ulta,
    df_sephora,
    how="outer",
    on=["product_lower", "brand_lower"],
    suffixes=("_ulta", "_sephora")
)

# Combine all columns, always prefer Sephora values
combined_cols = {}
for col in df_ulta.columns:
    if col in ["product_lower", "brand_lower"]:
        continue
    combined_cols[col] = df_merged[col + "_sephora"].combine_first(df_merged[col + "_ulta"])

# Keep original product and brand casing from Sephora if exists
combined_cols["product"] = df_merged["product_sephora"].combine_first(df_merged["product_ulta"])
combined_cols["brand"] = df_merged["brand_sephora"].combine_first(df_merged["brand_ulta"])

#Replace 'All skin types' with specific types for better accuracy
combined_cols["skin_type"] = combined_cols["skin_type"].replace({
    "All skin types": "normal, dry, oily, combination, sensitive"
})

# Create final DataFrame
df_combined = pd.DataFrame(combined_cols)

# Save to CSV
df_combined.to_csv("Full_Product_List.csv", index=False, encoding="utf-8-sig")
