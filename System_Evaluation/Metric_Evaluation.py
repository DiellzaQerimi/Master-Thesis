import argparse
import sys
import pandas as pd
import numpy as np

ALLOWED_VALUES = {0.0, 0.5, 1.0}

def require_cols(df: pd.DataFrame, cols: list[str]) -> None:
    # Validates that all required columns are present before running evaluation
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

def to_score(series: pd.Series, col: str) -> pd.Series:
    # Converts relevance labels into numeric scores and validates the allowed score set (0, 0.5, 1)
    s = series.copy()

    if s.dtype == bool:
        return s.astype(float)

    if s.dtype == object:
        t = s.astype(str).str.strip().str.lower()
        mapping = {
            "1": 1.0, "0": 0.0, "0.5": 0.5,
            "true": 1.0, "false": 0.0,
            "yes": 1.0, "no": 0.0,
            "relevant": 1.0, "irrelevant": 0.0,
        }
        mapped = t.map(mapping)
        out = pd.to_numeric(mapped.where(mapped.notna(), s), errors="coerce")
    else:
        out = pd.to_numeric(s, errors="coerce")

    if out.isna().any():
        bad = s[out.isna()].unique()[:10]
        raise ValueError(f"Column '{col}' has invalid values: {bad}")

    if not set(out.unique()).issubset(ALLOWED_VALUES):
        raise ValueError(f"Column '{col}' contains values outside {ALLOWED_VALUES}")

    return out.astype(float)

def mrr_at_k(group: pd.DataFrame, rank_col: str, rel_col: str) -> float:
    # Computes reciprocal rank for the first relevant item (relevance > 0) within the group
    rel = group[group[rel_col] > 0]
    if rel.empty:
        return 0.0
    return 1.0 / float(rel[rank_col].min())

def dcg_at_k(rels: np.ndarray) -> float:
    # Computes DCG using linear gains and log2 discounting for a ranked relevance list
    if rels.size == 0:
        return 0.0
    positions = np.arange(1, rels.size + 1, dtype=float)
    discounts = np.log2(positions + 1.0)
    return float(np.sum(rels / discounts))

def ndcg_at_k(group: pd.DataFrame, rank_col: str, rel_col: str, k: int) -> float:
    # Computes NDCG@K for a query by normalizing DCG by the ideal DCG for the same relevance set
    g = group.sort_values(rank_col, ascending=True)
    rels = g[rel_col].to_numpy(dtype=float)

    dcg = dcg_at_k(rels)

    ideal_rels = np.sort(rels)[::-1]
    idcg = dcg_at_k(ideal_rels)

    if idcg == 0.0:
        return 0.0
    return float(dcg / idcg)

def main():
    # Loads the input CSV, validates schema, computes Precision/Hit/MRR/NDCG at K per query, saves outputs, and prints overall averages
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", default="LMM_Automatic_Product_Evaluation.csv")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--out_prefix", default="Eval_Results")

    parser.add_argument("--query_col", default="query_id")
    parser.add_argument("--rank_col", default="rank")
    parser.add_argument("--relevant_col", default="overall_relevant_llm")

    parser.add_argument("--query_text_col", default="query_text")
    parser.add_argument("--brand_col", default="brand")
    parser.add_argument("--product_col", default="product_name")

    args = parser.parse_args()

    df = pd.read_csv(args.input)

    require_cols(
        df,
        [
            args.query_col,
            args.rank_col,
            args.relevant_col,
            args.query_text_col,
            args.brand_col,
            args.product_col,
        ]
    )

    df = df.copy()

    df[args.rank_col] = pd.to_numeric(df[args.rank_col], errors="coerce")
    df[args.relevant_col] = to_score(df[args.relevant_col], args.relevant_col)

    if df[args.rank_col].isna().any():
        raise ValueError(f"Column '{args.rank_col}' contains non-numeric ranks.")

    df_topk = df[df[args.rank_col] <= args.k].copy()

    per_query = (
        df_topk.groupby(args.query_col)
        .agg(
            query_text=(args.query_text_col, "first"),
            precision_at_k=(args.relevant_col, "mean"),
            sum_relevance=(args.relevant_col, "sum"),
        )
        .reset_index()
    )

    per_query["hit_at_k"] = (per_query["sum_relevance"] > 0).astype(int)

    mrr = (
        df_topk.groupby(args.query_col)
        .apply(lambda g: mrr_at_k(g, args.rank_col, args.relevant_col))
        .reset_index()
    )
    mrr.columns = [args.query_col, "mrr_at_k"]

    ndcg = (
        df_topk.groupby(args.query_col)
        .apply(lambda g: ndcg_at_k(g, args.rank_col, args.relevant_col, args.k))
        .reset_index()
    )
    ndcg.columns = [args.query_col, "ndcg_at_k"]

    per_query = per_query.merge(mrr, on=args.query_col, how="left")
    per_query = per_query.merge(ndcg, on=args.query_col, how="left")

    summary = {
        "K": args.k,
        "num_queries": int(per_query.shape[0]),
        "mean_precision_at_k": float(per_query["precision_at_k"].mean()),
        "mean_hit_at_k": float(per_query["hit_at_k"].mean()),
        "mean_mrr_at_k": float(per_query["mrr_at_k"].mean()),
        "mean_ndcg_at_k": float(per_query["ndcg_at_k"].mean()),
    }
    summary_df = pd.DataFrame([summary])

    per_query.to_csv(f"{args.out_prefix}_Per_Query.csv", index=False)
    summary_df.to_csv(f"{args.out_prefix}_Summary.csv", index=False)

    print("\n=== OVERALL SUMMARY ===")
    for k, v in summary.items():
        print(f"{k}: {v}")

    print("\nMetric evaluation completed")

if __name__ == "__main__":
    # Wraps execution with error handling so failures print cleanly and exit with non-zero status
    try:
        main()
    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        sys.exit(1)
