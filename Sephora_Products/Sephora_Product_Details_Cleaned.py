import re
import pandas as pd

# ---- Constants ----
ML_PER_OZ = 29.5735295625
text_col = 'about_the_product'

# ---- Utilities ----
def drop_if_contains(df, col, pattern):
    """Drop rows where `col` contains a given pattern (case-insensitive)."""
    return df.drop(df[df[col].str.contains(pattern, case=False, na=False)].index)

def clean_column(df, col, patterns):
    """Apply multiple regex replacements on a column."""
    for pat, repl in patterns:
        df[col] = df[col].replace(pat, repl, regex=True)
    return df


def clean_text_block(text, start_pat, stop_pat):
    """Extract text between start and stop headers."""
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


# ---- Load CSV File ----
df = pd.read_csv("Sephora_Products/Sephora_Product_Details.csv")

# ---- Cleaning ----
df = drop_if_contains(df, 'product', "perfume")
df = drop_if_contains(df, 'product', "product")
df = df[df['product_id'].astype(str).str.lower() != 'product_id']
df = df.replace(r'\n', ' ', regex=True)


# ---- Size Standardization ----
def standardize_size(s):
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

# ---- Description ----
df['description'] = df[text_col].apply(
    lambda x: clean_text_block(
        x,
        re.compile(r'(?i)(description|what\s+it\s+is)\s*:\s*'),
        re.compile(r'(?i)(what\s+else|(Skin\s+Type)\s*:|(Solutions\s+for)\s*:|formulated\s+WITHOUT|highlighted\s+ingredients\s*:|Ingredient\s+Callouts|skincare\s+concerns|show\s+less)')
    )
)

# ---- Important Ingredients extraction ----
df['important_ingredients'] = df[text_col].apply(
    lambda x: clean_text_block(
        x,
        re.compile(r'(?i)(highlighted\s+ingredients)\s*:\s*'),
        re.compile(r'(?i)(what\s+else|skin\s+type|formulation\s*:|fragrance\s+description\s*:|formulated\s+without\s*:|fragrance\s+family\s*:|(Ingredient\s+Callouts)\s*:|(Research\s+results)\s*:|skincare\s+concerns|clinical\s+results\s*:|show\s+less)')
    )
)

# ---- Skin Types ----
type_pat = re.compile(r'\b(oily|dry|combination|combo|normal)\b', re.I)

def get_skin_types(text):
    if not isinstance(text, str) or not text.strip():
        return pd.NA

    found = set()

    for m in type_pat.finditer(text):
        v = m.group(1).lower()

        # normalize combination
        if v in ("combo", "combination"):
            v = "combination"

        found.add(v)

    if not found:
        return pd.NA

    # If ALL 4 are present → All Skin Types
    required = {"normal", "dry", "combination", "oily"}
    if required.issubset(found):
        return "All Skin Types"

    # otherwise return the found ones
    return ", ".join(sorted([x.capitalize() for x in found]))


# ---- Skin Types Application ----
df['skin_type'] = df[text_col].apply(get_skin_types)

# ---- Skin Concerns ----
df['skin_concerns'] = df[text_col].apply(
    lambda x: clean_text_block(
        x,
        re.compile(r'(?i)(skincare\s*concerns|solutions\s*\+?\s*for)\s*:\s*'),
        re.compile(r'(?i)(what\s*else|skin\s*type|formulation\s*:|ingredient\s+callouts\s*:|shade\s+description\s*:|highlighted\s+ingredients\s*:|fragrance\s+description\s*:|fragrance\s+family\s*:|show\s*less|if\s*you\s*want)')
    )
)

# ---- Size Standardization Application ----
df['size'] = df['size'].map(standardize_size)
# ---- Skin Concerns Normalization ----
df['skin_concerns'] = df['skin_concerns'].str.title()

# ---- Free Of Extraction ----
df['free_of'] = df[text_col].apply(
    lambda x: clean_text_block(
        x,
        re.compile(r'(?i)(ingredient\s+callouts|formulated\s+WITHOUT)\s*:\s*'),
        re.compile(r'(?i)(what\s+else|skin\s+type|formulation\s*:|skincare\s+concerns|research\s+results|show\s+less)')
    )
)

# no_of_reviews Cleaning
df = clean_column(df, 'no_of_reviews', [
    (r'(?i)^\s*write\s+a\s+review\s*$', pd.NA)
])

# how_to_use Cleaning
df = clean_column(df, 'how_to_use', [
    (r'(?i)suggested\s+usage\s*:', ''),  
    ('•', '-')  
])

# description Cleaning
df = clean_column(df, 'description', [
    (r'(?i)\bwhat\s+it\s+is\s+formulated\s+to\s+do\s*:\s*', '')
])

# skin_concerns Cleaning
df = clean_column(df, 'skin_concerns', [
    (r'-', '')  # remove stray dashes
])

df['source'] = "Sephora"

pd.set_option('display.max_colwidth', None)  # don’t cut off long text
df.to_csv("Sephora_Products/Sephora_Product_Details_Cleaned.csv", index=False, encoding="utf-8-sig")

id_ = "P510508"
result = {
    "description": df.loc[df['product_id'] == id_, 'description'].head(3).tolist(),
    "skin_type": df.loc[df['product_id'] == id_, 'skin_type'].head(3).tolist(),
    "free_of": df.loc[df['product_id'] == id_, 'free_of'].head(3).tolist(),
    # "ingredients": df.loc[df['product_id'] == id_, 'ingredients'].head(3).tolist(),
    "important_ingredients": df.loc[df['product_id'] == id_, 'important_ingredients'].head(3).tolist(),
}
print(result)