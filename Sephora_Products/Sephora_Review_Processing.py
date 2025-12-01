import pandas as pd
import datetime as date

# Load datasets
df_sephora = pd.read_csv("Sephora_Products/Sephora_Product_Details.csv", low_memory=False)
df_reviews = pd.read_csv("Sephora_Products/Init_Sephora_Product_Reviews.csv", low_memory=False)

# Keep only product_id + product name
df_sephora_small = df_sephora[["product_id", "product", "brand"]].drop_duplicates()

# Merge reviews with product names
df_inner = pd.merge(
    df_sephora_small,
    df_reviews,
    how="inner",
    on="product_id"
)
# Convert timestamp to datetime if needed
df_inner['submission_time'] = pd.to_datetime(df_inner['submission_time'])

# Function to classify season
def classify_season(date):
    month = date.month
    
    if month in [10, 11, 12, 1, 2, 3]:
        return "Winter"
    elif month in [4, 5, 6, 7, 8, 9]:
        return "Summer"
    else:
        return "Other"  # Apr, May

columns_to_drop = ["skin_type", "skin_tone", "age", "eye_color", "hair_color", "product_name"]
df_combined = df_inner.drop(columns=columns_to_drop, errors='ignore')

# Apply to DataFrame
df_combined['season_category'] = df_combined['submission_time'].apply(classify_season)

# Add source column
df_combined['source'] = 'Sephora'

# Save to CSV
df_combined.to_csv("Sephora_Product_Reviews.csv", index=False, encoding="utf-8-sig")
