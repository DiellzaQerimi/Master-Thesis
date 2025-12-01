import pandas as pd
import datetime as date

# Load datasets
df_ulta = pd.read_csv("Ulta_Products/Ulta_Product_Details.csv", low_memory=False)
df_reviews = pd.read_csv("Ulta_Products/Init_Ulta_Product_Reviews.csv", low_memory=False)

# Keep only product_id + product name
df_ulta_small = df_ulta[["product_id", "product", "brand"]].drop_duplicates()

# Merge reviews with product names
df_inner = pd.merge(
    df_ulta_small,
    df_reviews,
    how="inner",
    on="product_id"
)

# Clean review text
df_inner["review_text"] = df_inner["review_text"].replace(
    r'\[I received this product in exchange for my honest review\]',
    '', 
    regex=True
)
# Convert timestamp to datetime if needed
df_inner['submission_time'] = pd.to_datetime(df_inner['submission_time'], unit='ms')

# Function to classify season
def classify_season(date):
    month = date.month
    
    if month in [10, 11, 12, 1, 2, 3]:
        return "Winter"
    elif month in [4, 5, 6, 7, 8, 9]:
        return "Summer"
    else:
        return "Other"  # Apr, May


df_inner = df_inner.drop(columns=['location'])
# Apply to DataFrame
df_inner['season_category'] = df_inner['submission_time'].apply(classify_season)

# Add source column
df_inner['source'] = 'Ulta Beauty'

# Save to CSV
df_inner.to_csv("Ulta_Product_Reviews.csv", index=False, encoding="utf-8-sig")