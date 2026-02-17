import pandas as pd
import datetime as date

# Loads product details and review datasets, merges them by product_id, adds seasonal labels, and exports a cleaned reviews file
df_ulta = pd.read_csv("Ulta_Products/Ulta_Product_Details.csv", low_memory=False)
df_reviews = pd.read_csv("Init_Ulta_Product_Reviews.csv", low_memory=False)

# Keeps only the product fields needed for joining (product_id, product, brand)
df_ulta_small = df_ulta[["product_id", "product", "brand"]].drop_duplicates()

# Merges review records with product details using product_id as the join key
df_inner = pd.merge(
    df_ulta_small,
    df_reviews,
    how="inner",
    on="product_id"
)

# Removes the standard incentivized-review disclaimer text from review_text
df_inner["review_text"] = df_inner["review_text"].replace(
    r'\[I received this product in exchange for my honest review\]',
    '', 
    regex=True
)

# Converts submission_time into datetime format for time-based transformations
df_inner['submission_time'] = pd.to_datetime(df_inner['submission_time'], unit='ms')

# Classifies review timestamps into seasonal categories based on the month value
def classify_season(date):
    month = date.month
    
    if month in [10, 11, 12, 1, 2, 3]:
        return "Winter"
    elif month in [4, 5, 6, 7, 8, 9]:
        return "Summer"
    else:
        return "Other"  # Apr, May

# Drops location field to keep the final dataset focused
df_inner = df_inner.drop(columns=['location'])

# Applies seasonal classification to each review record
df_inner['season_category'] = df_inner['submission_time'].apply(classify_season)

# Adds a source flag to identify where the reviews were collected from
df_inner['source'] = 'Ulta Beauty'

# Saves the final merged and labeled reviews dataset to CSV
df_inner.to_csv("Ulta_Product_Reviews.csv", index=False, encoding="utf-8-sig")
