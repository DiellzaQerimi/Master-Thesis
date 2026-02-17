import os
import time
import argparse
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

ALLOWED = {"0", "0.5", "1"}

# Helpers
def build_product_id(row, brand_col, product_col):
    # Builds a consistent product_id using the same brand+product normalization as the embedding pipeline
    return f"{str(row[brand_col]).strip().lower()} {str(row[product_col]).strip().lower()}"

def pick_about_the_product(row, cols):
    # Selects the first available description field from the candidate columns
    for c in cols:
        if c in row and pd.notna(row[c]) and str(row[c]).strip():
            return str(row[c]).strip()
    return "(missing)"

def build_prompt(query, brand, product, category, subcategory, price, about_the_product):
    # Builds a strict prompt that forces the LLM to output only a single allowed relevance score
    return f"""
You are evaluating skincare product recommendations.

User query:
{query}

Recommended product:
- Brand: {brand}
- Name: {product}
- Category: {category} / {subcategory}
- Price: {price}

Product about_the_product:
{about_the_product}

Task:
Rate how relevant this product is for the user query.

Scoring:
1 = fully relevant
0.5 = partially relevant
0 = not relevant

Return ONLY one of: 0, 0.5, 1
""".strip()

def llm_score(client, model, prompt):
    # Sends the prompt to the LLM and extracts a numeric relevance score (0, 0.5, 1) from the response
    resp = client.responses.create(
        model=model,
        input=prompt,
        temperature=0
    )

    text = resp.output_text.strip()
    token = (
        text.replace("Score:", "")
            .replace("score:", "")
            .strip()
            .split()[0]
            .rstrip(".")
            .rstrip(",")
    )

    if token in ALLOWED:
        return float(token)

    for cand in ["0.5", "1", "0"]:
        if cand in text:
            return float(cand)

    return 0.0

# Main
def main():
    # Loads inputs, creates product_id for the product list, merges descriptions internally, runs LLM scoring, and exports a clean evaluation CSV
    parser = argparse.ArgumentParser()

    parser.add_argument("--recs_csv", default="../Product_Recommendation_List.csv")
    parser.add_argument("--products_csv", default="../Full_Product_List.csv")
    parser.add_argument("--out_csv", default="LLM_Automatic_Product_Evaluation.csv")

    parser.add_argument("--query_col", default="query_text")
    parser.add_argument("--brand_col", default="brand")
    parser.add_argument("--product_col", default="product_name")
    parser.add_argument("--category_col", default="category")
    parser.add_argument("--subcategory_col", default="subcategory")
    parser.add_argument("--price_col", default="price_usd")

    parser.add_argument("--prod_brand_col", default="brand")
    parser.add_argument("--prod_product_col", default="product")
    parser.add_argument("--desc_candidates", default="about_the_product")

    parser.add_argument("--model", default="gpt-4.1-mini")
    parser.add_argument("--sleep", type=float, default=0.05)

    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not set")

    client = OpenAI()

    recs = pd.read_csv(args.recs_csv)
    products = pd.read_csv(args.products_csv)

    products = products.copy()
    products["product_id"] = products.apply(
        lambda r: build_product_id(r, args.prod_brand_col, args.prod_product_col),
        axis=1
    )

    products = products.drop_duplicates(subset=["product_id"], keep="first")

    merged = recs.merge(products, on="product_id", how="left", suffixes=("", "_prod"))

    desc_cols = [c.strip() for c in args.desc_candidates.split(",")]

    scores = []

    for _, row in merged.iterrows():
        about_the_product = pick_about_the_product(row, desc_cols)

        prompt = build_prompt(
            query=str(row.get(args.query_col, "")),
            brand=str(row.get(args.brand_col, "")),
            product=str(row.get(args.product_col, "")),
            category=str(row.get(args.category_col, "")),
            subcategory=str(row.get(args.subcategory_col, "")),
            price=str(row.get(args.price_col, "")),
            about_the_product=about_the_product,
        )

        score = llm_score(client, args.model, prompt)
        scores.append(score)
        time.sleep(args.sleep)

    merged["overall_relevant_llm"] = scores

    out_cols = list(recs.columns) + ["overall_relevant_llm"]
    out_df = merged[out_cols].copy()

    if "query_id" in out_df.columns:
        out_df["eval_query_id"] = out_df["query_id"].astype(str)
    else:
        out_df["eval_query_id"] = out_df[args.query_col].astype(str)

    out_df = out_df.loc[:, ~out_df.columns.str.contains("^Unnamed")]

    out_df.to_csv(args.out_csv, index=False, encoding="utf-8-sig")

    print("Automatic evaluation completed")
    print(f"Saved {args.out_csv}")

if __name__ == "__main__":
    main()
