import pandas as pd
import datetime as date

# Loads product details and review datasets, merges them by product_id, adds seasonal labels, and exports a cleaned reviews file
df_sephora = pd.read_csv("Sephora_Products/Sephora_Product_Details.csv", low_memory=False)
df_reviews = pd.read_csv("Init_Sephora_Product_Reviews.csv", low_memory=False)

# Keeps only the product fields needed for joining (product_id, product, brand)
df_sephora_small = df_sephora[["product_id", "product", "brand"]].drop_duplicates()

# Merges review records with product details using product_id as the join key
df_inner = pd.merge(
    df_sephora_small,
    df_reviews,
    how="inner",
    on="product_id"
)

# Converts submission_time into datetime format for time-based transformations
df_inner['submission_time'] = pd.to_datetime(df_inner['submission_time'])

# Classifies review timestamps into seasonal categories based on the month value
def classify_season(date):
    month = date.month
    
    if month in [10, 11, 12, 1, 2, 3]:
        return "Winter"
    elif month in [4, 5, 6, 7, 8, 9]:
        return "Summer"
    else:
        return "Other"  # Apr, May

# Drops unused review-related columns to keep the final dataset focused
columns_to_drop = ["eye_color", "hair_color", "product_name"]
df_combined = df_inner.drop(columns=columns_to_drop, errors='ignore')

# Applies seasonal classification to each review record
df_combined['season_category'] = df_combined['submission_time'].apply(classify_season)

# Adds a source flag to identify where the reviews were collected from
df_combined['source'] = 'Sephora'

# Saves the final merged and labeled reviews dataset to CSV
df_combined.to_csv("Sephora_Product_Reviews.csv", index=False, encoding="utf-8-sig")
