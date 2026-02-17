import pandas as pd

# Load manual evaluation file
df = pd.read_csv("Manual_Product_Evaluation.csv")

# Define relevance columns
columns = {
    "Ingredient Match": "ingredient_match",
    "Skin Type Match": "skin_type_match",
    "Concern Match": "concern_match",
    "Category Match": "category_match",
    "Overall Relevance": "overall_relevance"
}

# Compute mean per dimension (ignores NaN automatically)
relevance_table = pd.DataFrame({
    "Relevance Dimension": columns.keys(),
    "Mean Score": [df[col].mean() for col in columns.values()]
})

# Round for thesis formatting
relevance_table["Mean Score"] = relevance_table["Mean Score"].round(2)

print(relevance_table)
