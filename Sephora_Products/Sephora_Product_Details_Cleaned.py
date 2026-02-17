import re
import pandas as pd

# Conversion constant used for size standardization (mL to oz)
ML_PER_OZ = 29.5735295625

# Main text column containing product information blocks
text_col = 'about_the_product'

def drop_if_contains(df, col, pattern):
    # Removes rows where the specified column contains a given pattern
    return df.drop(df[df[col].str.contains(pattern, case=False, na=False)].index)

def clean_column(df, col, patterns):
    # Applies multiple regex-based cleaning rules to a specific column
    for pat, repl in patterns:
        df[col] = df[col].replace(pat, repl, regex=True)
    return df

def clean_text_block(text, start_pat, stop_pat):
    # Extracts structured text sections between defined start and stop headers
    if not isinstance(text, str) or not text.strip():
        return pd.NA

    hits = []
    for m in start_pat.finditer(text):
        tail = text[m.end():]
        stop_match = stop_pat.search(tail)
        if stop_match:
            tail = tail[:stop_match.start()]

        tail_norm = re.sub(r'\s+', ' ', tail).strip()
        if tail_norm:
            hits.append(tail_norm)

    if not hits:
        return pd.NA

    seen, out = set(), []
    for h in hits:
        k = h.casefold()
        if k not in seen:
            seen.add(k)
            out.append(h)

    return ' '.join(out)

# Load raw Sephora product details dataset
df = pd.read_csv("Sephora_Products/Sephora_Product_Details.csv")

# Remove unwanted rows and normalize newline formatting
df = drop_if_contains(df, 'product', "perfume")
df = drop_if_contains(df, 'product', "product")
df = df[df['product_id'].astype(str).str.lower() != 'product_id']
df = df.replace(r'\n', ' ', regex=True)

def standardize_size(s):
    # Converts size representations into standardized "oz / mL" format
    if not isinstance(s, str):
        return s
    t = re.sub(r'(?i)^\s*size\b\s*:?\s*', '', s.strip())
    t = re.sub(r'\s*/\s*', ' / ', t)

    m = re.search(r'(\d+(?:\.\d+)?)\s*ml\b', t, flags=re.I)
    o = re.search(r'(\d+(?:\.\d+)?)\s*(?:fl\.?\s*)?oz\b', t, flags=re.I)
    ml = float(m.group(1)) if m else None
    oz = float(o.group(1)) if o else None

    if ml is None and oz is not None:
        ml = oz * ML_PER_OZ
    if oz is None and ml is not None:
        oz = ml / ML_PER_OZ

    return f"{oz:.2f} oz / {ml:.0f} mL" if ml and oz else t

# Extract product description section from text column
df['description'] = df[text_col].apply(
    lambda x: clean_text_block(
        x,
        re.compile(r'(?i)(description|what\s+it\s+is)\s*:\s*'),
        re.compile(r'(?i)(what\s+else|(Skin\s+Type)\s*:|(Solutions\s+for)\s*:|formulated\s+WITHOUT|highlighted\s+ingredients\s*:|Ingredient\s+Callouts|skincare\s+concerns|show\s+less)')
    )
)

# Extract highlighted / important ingredients section
df['important_ingredients'] = df[text_col].apply(
    lambda x: clean_text_block(
        x,
        re.compile(r'(?i)(highlighted\s+ingredients)\s*:\s*'),
        re.compile(r'(?i)(what\s+else|skin\s+type|formulation\s*:|fragrance\s+description\s*:|formulated\s+without\s*:|fragrance\s+family\s*:|(Ingredient\s+Callouts)\s*:|(Research\s+results)\s*:|skincare\s+concerns|clinical\s+results\s*:|show\s+less)')
    )
)

# Pattern used to identify skin types inside product text
type_pat = re.compile(r'\b(oily|dry|combination|combo|normal)\b', re.I)

def get_skin_types(text):
    # Detects and normalizes skin type mentions from text
    if not isinstance(text, str) or not text.strip():
        return pd.NA

    found = set()

    for m in type_pat.finditer(text):
        v = m.group(1).lower()
        if v in ("combo", "combination"):
            v = "combination"
        found.add(v)

    if not found:
        return pd.NA

    return ", ".join(sorted([x.capitalize() for x in found]))

# Apply skin type extraction
df['skin_type'] = df[text_col].apply(get_skin_types)

# Extract skincare concerns section
df['skin_concerns'] = df[text_col].apply(
    lambda x: clean_text_block(
        x,
        re.compile(r'(?i)(skincare\s*concerns|solutions\s*\+?\s*for)\s*:\s*'),
        re.compile(r'(?i)(what\s*else|skin\s*type|formulation\s*:|ingredient\s+callouts\s*:|shade\s+description\s*:|highlighted\s+ingredients\s*:|fragrance\s+description\s*:|fragrance\s+family\s*:|show\s*less|if\s*you\s*want)')
    )
)

# Apply size standardization
df['size'] = df['size'].map(standardize_size)

# Normalize skin concerns formatting
df['skin_concerns'] = df['skin_concerns'].str.title()

# Extract "free of" / formulated without section
df['free_of'] = df[text_col].apply(
    lambda x: clean_text_block(
        x,
        re.compile(r'(?i)(ingredient\s+callouts|formulated\s+WITHOUT)\s*:\s*'),
        re.compile(r'(?i)(what\s+else|skin\s+type|formulation\s*:|skincare\s+concerns|research\s+results|show\s+less)')
    )
)

# Clean no_of_reviews column
df = clean_column(df, 'no_of_reviews', [
    (r'(?i)^\s*write\s+a\s+review\s*$', pd.NA)
])

# Clean how_to_use column
df = clean_column(df, 'how_to_use', [
    (r'(?i)suggested\s+usage\s*:', ''),  
    ('•', '-')  
])

# Clean description column
df = clean_column(df, 'description', [
    (r'(?i)\bwhat\s+it\s+is\s+formulated\s+to\s+do\s*:\s*', '')
])

# Clean skin_concerns column
df = clean_column(df, 'skin_concerns', [
    (r'-', '')
])

# Add source identifier column
df['source'] = "Sephora"

# Ensure full text display in console
pd.set_option('display.max_colwidth', None)

# Save cleaned dataset to new CSV file
df.to_csv("Sephora_Products/Sephora_Product_Details_Cleaned.csv", index=False, encoding="utf-8-sig")

# Test extraction for a specific product ID
id_ = "P510508"
result = {
    "description": df.loc[df['product_id'] == id_, 'description'].head(3).tolist(),
    "skin_type": df.loc[df['product_id'] == id_, 'skin_type'].head(3).tolist(),
    "free_of": df.loc[df['product_id'] == id_, 'free_of'].head(3).tolist(),
    # "ingredients": df.loc[df['product_id'] == id_, 'ingredients'].head(3).tolist(),
    "important_ingredients": df.loc[df['product_id'] == id_, 'important_ingredients'].head(3).tolist(),
}
print(result)
