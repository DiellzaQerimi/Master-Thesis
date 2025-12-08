import re
import pandas as pd

# ---- Load CSV File ----
df = pd.read_csv("Ulta_Products/Ulta_Product_Details.csv", low_memory=False)
text_col = "about_the_product"

# ---- Constants ----
ML_PER_OZ = 29.5735295625

# ---- Standardize Sizes ----
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

    return f"{oz:.2f} oz / {ml:.0f} mL" if (ml is not None and oz is not None) else t

#---- Text Extraction Functions ----
def clean_text_block(text, start_pat, stop_pat):
    """
    Extract text after a start header until a stop header.
    Handles deduplication, whitespace normalization.
    """
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

    # De-dup, preserve order
    seen, out = set(), []
    for h in hits:
        k = h.casefold()
        if k not in seen:
            seen.add(k)
            out.append(h)

    return ' '.join(out)

# Description block
df['description'] = df[text_col].apply(
    lambda x: clean_text_block(
        x,
        re.compile(r'^', flags=re.S),  # start of text, dot matches newlines
        re.compile(
            r'((?:–|-)\s*benefits|(?:–|-)\s*key\s+ingredients|benefits\s*|key\s+ingredients)'
            r'|((?:–|-)\s*features|features|research\s+results\s*)|item\s*|formulated\s+without\s*|Item\s*|skin\s+type',
            flags=re.I
        )
    )
)

# Skin Type extraction
def get_skin_type(text):
    if not isinstance(text, str) or not text.strip():
        return pd.NA

    val = clean_text_block(
        text,
        re.compile(r'(?i)(skin\s+type\s*:)'),
        re.compile(
            r'(?i)(?=\.|skin\s*concerns|ingredient\s+callouts\s*|skincare\s*concerns|(research\s+results)\s*|features\s*)'
        )
    )

    if pd.isna(val):
        pattern = re.compile(
            r'(?i)\b('
            r'(?:dry|oily|normal|sensitive|combination)'
            r'(?:[\s,&]*(?:and|to)?[\s,&]*(?:dry|oily|normal|sensitive|combination))*'
            r'\s+skin(?:\s*types?)?|'  
            r'all\s+skin\s+types?'
            r')\b'
        )
        matches = pattern.findall(text)
        if matches:
            val = " ".join(matches)
        else:
            val = pd.NA

    if isinstance(val, str) and val.strip():
        type_words = re.findall(
            r'\b(dry|oily|normal|sensitive|combination|all\s+skin\s+types)\b',
            val,
            flags=re.I
        )

        order = ['Dry', 'Oily', 'Normal', 'Sensitive', 'Combination', 'All Skin Types']
        type_clean = sorted(
            {t.title() for t in type_words},
            key=lambda x: order.index(x) if x in order else 999
        )

        if 'All Skin Types' in type_clean and len(type_clean) > 1:
            return 'All Skin Types'

        return ", ".join(type_clean) if type_clean else pd.NA

    return val

# Skin Concerns extraction
concern_pat = re.compile(
    r"\b("
    r"fine\s+lines?\s*(?:and|&)?\s*wrinkles?|wrinkles?|fine[\s-]+line[s]?|"
    r"acne|blemishes?|redness|pores?|"
    r"dull|dullness|cracked|chapped|"
    r"uneven\s+tone|dark\s+circles?|dark\s+spots?|hyperpigmentation|blackheads?|dryness|"
    r"(?:uneven|rough|improve|refine|smooth)\s+texture|"
    r"oiliness|sensitivity|aging|firmness|elasticity|"
    r"puffiness|depuff(?:ing|ed|s)?|sagging|saggy"
    r")\b",
    re.IGNORECASE
)

# Extract and standardize skin concerns
def get_skin_concerns(text):
    """Extract and standardize skin concerns from product descriptions."""
    if not isinstance(text, str) or not text.strip():
        return pd.NA
    
    found = []
    for m in concern_pat.finditer(text):
        v = m.group(1).lower()

        # Normalize terms
        if v.startswith('fine line') or v.startswith('wrinkle'):
            v = 'fine lines & wrinkles'
        elif v in ('blemish', 'acne'):
            v = 'acne & blemishes'
        elif v in ('dark spot', 'hyperpigmentation'):
            v = 'dark spots'
        elif v.startswith('dark circles'):
            v = 'dark circles'
        elif v.startswith('depuff') or v == 'puffiness':
            v = 'puffiness'
        elif v in ('cracked', 'chapped'):
            v = 'dryness'
        elif v.startswith('dull'):
            v = 'dullness'
        elif v in ('pore', 'pores'):
            v = 'pores'
        elif v in ('saggy', 'sagging'):
            v = 'sagging'
        elif v in ('improve texture', 'refine texture', 'smooth texture', 'uneven texture', 'texture'):
            v = 'texture'
        elif v in ('oiliness', 'dryness', 'redness'):
            v = v
        else:
            v = v
        
        label = v.capitalize()
        if label not in found:
            found.append(label)
    
    return ', '.join(found) if found else pd.NA

# Formulated Without / Free Of extraction
def get_formulated_without(text):
    if not isinstance(text, str) or not text.strip():
        return pd.NA

    val = clean_text_block(
        text,
        re.compile(r'(?i)(formulated\s+without\s*|ingredient\s+callouts\s*:|made\s+without\s*:|free\s+of\s*)'),
        re.compile(r'(?i)(?=\.|always\s+formulated\s+without\s*|item\s*|research\s+results\s*|features\s*|includes\s*|dermatologist\s*|skin\s+type\s*|key\s+ingredients\s*)')
    )

    if pd.isna(val):
        pattern = re.compile(
            r'(?i)(paraben|fragrance|sulfate|oil|alcohol|gluten|cruelty|phthalate)[-\s]*free'
        )
        matches = pattern.findall(text)

        if matches:
            # Normalize to the clean hyphenated form: X-Free
            normalized = sorted(set(f"{m.title()}-Free" for m in matches))
            val = ", ".join(normalized)
        else:
            val = pd.NA

    return val

# Important Ingredients extraction
def extract_ingredients(text):
    val = clean_text_block(
        text,
        re.compile(r'(?i)key\s+ingredients\s*'),
        re.compile(r'(?i)(formulated\s+without\s*|Item\s*|skin\s+type|(research\s+results)\s*|features\s*)')
    )
    if pd.isna(val):
        val = clean_text_block(
            text,
            re.compile(r'(?i)Features\s*'),
            re.compile(r'(?i)(Benefits|formulated\s+without|Item|research\s+results|skincare\s+concerns|skin\s+type)\s*')
        )
    return val


# ---- Apply Cleaning Functions ----
df['size'] = df['size'].map(standardize_size)
df['important_ingredients'] = df[text_col].apply(extract_ingredients)
df['skin_type'] = df[text_col].apply(get_skin_type)
df['skin_concerns'] = df[text_col].apply(get_skin_concerns).apply(lambda x: x.title() if isinstance(x, str) else x)
df['free_of'] = df[text_col].apply(get_formulated_without)
df['free_of'] = df['free_of'].str.replace(r' / ', ', ', regex=True)
df['source'] = "Ulta Beauty"

# Sanity check for the product in user's snippet
id_ = "pimprod2051662"
result = {
    "description": df.loc[df['product_id'] == id_, 'description'].head(3).tolist(),
    "skin_type": df.loc[df['product_id'] == id_, 'skin_type'].head(3).tolist(),
    "free_of": df.loc[df['product_id'] == id_, 'free_of'].head(3).tolist(),
    # "ingredients": df.loc[df['product_id'] == id_, 'ingredients'].head(3).tolist(),
    "important_ingredients": df.loc[df['product_id'] == id_, 'important_ingredients'].head(3).tolist(),
}

# Save cleaned
out_path = "Ulta_Products/Ulta_Product_Details_Cleaned.csv"
df.to_csv(out_path, index=False, encoding="utf-8-sig")
print(result)
# result, out_path