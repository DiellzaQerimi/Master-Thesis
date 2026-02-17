# SearchQuery.py
import os
import re
import csv
import json
import unicodedata
from datetime import datetime
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue

# =========================================================
# CONFIG
# =========================================================
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
qdrant = QdrantClient(url="http://localhost:6333", timeout=180)

PRODUCTS_COL = "product_list"
REVIEWS_COL = "product_reviews"

EMBED_MODEL = "text-embedding-3-small"
REVIEW_SUMMARY_MODEL = os.getenv("REVIEW_SUMMARY_MODEL", "gpt-4o-mini")

TOP_K_PRODUCTS = 20
TOP_N_PRODUCTS = 10
TOP_DEEP_PRODUCTS = 3

REVIEWS_STAGE_A = 20
REVIEWS_STAGE_B = 50

# ---- payload keys (product_list) ----
PRODUCT_ID_KEY = "product_id"
BRAND_DISPLAY_KEY = "brand_name"
BRAND_NORM_KEY = "norm_brand"
NAME_KEY = "product_name"
PRICE_KEY = "price"

CATEGORY_KEY = "category"
SUBCATEGORY_KEY = "subcategory"
SIZE_KEY = "size"
IMAGE_URL_KEY = "image_url"
SKIN_TYPE_KEY = "skin_type"
SKIN_CONCERN_KEY = "skin_concerns"
DESCRIPTION_KEY = "description"
IMPORTANT_INGREDIENTS_KEY = "important_ingredients"
HOW_TO_USE_KEY = "how_to_use"

# ---- payload keys (product_reviews) ----
REVIEW_JOIN_KEY = "brand_product_id"  # join field in reviews payload
REVIEW_TEXT_KEY = "review_text"
REVIEW_ID_KEY = "review_id"
REVIEW_RATING_KEY = "rating"
REVIEW_DATE_KEY = "review_date"

# =========================================================
# CSV LOGGER (FOR MANUAL EVALUATION)
# =========================================================
CSV_PATH = "Automatic_Evaluation_Recommendations2.csv"


def log_recommendations_to_csv(
    query_id: int,
    query_text: str,
    stage_a: List[Dict[str, Any]],
    stage_b: List[Dict[str, Any]],
    extracted_brand_norm: Optional[str],
    used_brand_filter: bool,
    price_intent: Optional[Dict[str, float]],
) -> None:
    """
    Appends recommendations to a CSV file.
    - One row per recommendation
    - Logs top 5 from Stage A and all Stage B (usually top 3)
    """
    file_exists = os.path.isfile(CSV_PATH)

    with open(CSV_PATH, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow(
                [
                    "run_timestamp_utc",
                    "query_id",
                    "query_text",
                    "rank",
                    "stage",
                    "product_id",
                    "brand",
                    "product_name",
                    "category",
                    "subcategory",
                    "price_usd",
                    "final_score",
                    "embedding_score",
                    "review_avg",
                    "reviews_used",
                    "extracted_brand_norm",
                    "used_brand_filter",
                    "price_intent_json",
                ]
            )

        ts = datetime.utcnow().isoformat()
        price_intent_json = json.dumps(price_intent) if price_intent else ""

        # Stage A (top 5)
        for i, p in enumerate(stage_a[:5], 1):
            writer.writerow(
                [
                    ts,
                    query_id,
                    query_text,
                    i,
                    "stage_a",
                    p.get("product_id"),
                    p.get("brand"),
                    p.get("name"),
                    p.get("category"),
                    p.get("subcategory"),
                    p.get("price"),
                    round(float(p.get("final_score", 0.0) or 0.0), 6),
                    round(float(p.get("score", 0.0) or 0.0), 6),
                    round(float(p.get("review_avg", 0.0) or 0.0), 6),
                    int(p.get("reviews_used", 0) or 0),
                    extracted_brand_norm or "",
                    "1" if used_brand_filter else "0",
                    price_intent_json,
                ]
            )

        # Stage B (top deep products, usually 3)
        for i, p in enumerate(stage_b, 1):
            writer.writerow(
                [
                    ts,
                    query_id,
                    query_text,
                    i,
                    "stage_b",
                    p.get("product_id"),
                    p.get("brand"),
                    p.get("name"),
                    p.get("category"),
                    p.get("subcategory"),
                    p.get("price"),
                    round(float(p.get("final_score", 0.0) or 0.0), 6),
                    round(float(p.get("score", 0.0) or 0.0), 6),
                    round(float(p.get("review_avg", 0.0) or 0.0), 6),
                    int(p.get("reviews_used", 0) or 0),
                    extracted_brand_norm or "",
                    "1" if used_brand_filter else "0",
                    price_intent_json,
                ]
            )


# =========================================================
# HELPERS
# =========================================================
def embed(text: str) -> List[float]:
    r = client.embeddings.create(model=EMBED_MODEL, input=text)
    return r.data[0].embedding


def norm_brand(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s


def strip_brand_phrase(user_query: str) -> str:
    q = user_query

    q = re.sub(
        r"\bfrom\b\s+([A-Za-z0-9][A-Za-z0-9\-'\s]{1,80})"
        r"(?=\s+\b(under|below|max|less than|over|above|more than|between|"
        r"drugstore|high end|luxury|premium|budget|affordable|"
        r"for|with|without|on|in)\b|\s*$)",
        "",
        q,
        flags=re.IGNORECASE,
    )

    q = re.sub(
        r"\bby\b\s+([A-Za-z0-9][A-Za-z0-9\-'\s]{1,80})"
        r"(?=\s+\b(under|below|max|less than|over|above|more than|between|"
        r"drugstore|high end|luxury|premium|budget|affordable|"
        r"for|with|without|on|in)\b|\s*$)",
        "",
        q,
        flags=re.IGNORECASE,
    )

    q = re.sub(r"\bfrom\b\s+([A-Za-z0-9\-'\s]{1,80})\s*$", "", q, flags=re.IGNORECASE)
    q = re.sub(r"\bby\b\s+([A-Za-z0-9\-'\s]{1,80})\s*$", "", q, flags=re.IGNORECASE)

    q = re.sub(r"\s{2,}", " ", q).strip()
    return q


def extract_brand(user_query: str) -> Optional[str]:
    q = user_query.strip()

    m = re.search(
        r"\b(from|by)\b\s+([A-Za-z0-9][A-Za-z0-9\-'\s]{1,80})",
        q,
        flags=re.IGNORECASE,
    )
    if not m:
        return None

    tail = m.group(2).strip()

    stop_at = re.split(
        r"\b(under|below|max|less than|over|above|more than|between|"
        r"drugstore|high end|luxury|premium|budget|affordable|expensive|"
        r"for|with|without|on|in)\b",
        tail,
        flags=re.IGNORECASE,
        maxsplit=1,
    )[0].strip()

    words = re.findall(r"[A-Za-z0-9\-']+", stop_at)
    if not words:
        return None

    brand_phrase = " ".join(words[:4])
    bn = norm_brand(brand_phrase)
    return bn or None


def price_str_to_float(price_str: Optional[str]) -> Optional[float]:
    if not price_str:
        return None
    m = re.search(r"(\d+(?:\.\d+)?)", str(price_str).replace(",", ""))
    return float(m.group(1)) if m else None


def resolve_price_intent(q: str) -> Optional[Dict[str, float]]:
    q = q.lower()
    if m := re.search(r"(under|below|max|less than)\s*\$?\s*(\d+(?:\.\d+)?)", q):
        return {"lte": float(m.group(2))}
    if m := re.search(r"(over|above|more than)\s*\$?\s*(\d+(?:\.\d+)?)", q):
        return {"gte": float(m.group(2))}
    if m := re.search(r"between\s*\$?\s*(\d+(?:\.\d+)?)\s*(and|-)\s*\$?\s*(\d+(?:\.\d+)?)", q):
        lo, hi = float(m.group(1)), float(m.group(3))
        return {"gte": min(lo, hi), "lte": max(lo, hi)}
    if "drugstore" in q or "affordable" in q:
        return {"lte": 50.0}
    if "luxury" in q or "high end" in q or "premium" in q or "expensive" in q:
        return {"gte": 50.0}

    # Soft price language (you mentioned: "not too expensive")
    # For now we treat it as a soft cap to support your tests.
    if "not too expensive" in q or "prefer not too expensive" in q or "not expensive" in q:
        return {"lte": 30.0}

    return None


def apply_price_filter(products: List[Dict[str, Any]], price_intent: Dict[str, float]) -> List[Dict[str, Any]]:
    out = []
    for p in products:
        price = price_str_to_float(p.get("price"))
        if price is None:
            continue
        if "gte" in price_intent and price < price_intent["gte"]:
            continue
        if "lte" in price_intent and price > price_intent["lte"]:
            continue
        p["price_num"] = price
        out.append(p)
    return out


# =========================================================
# PRODUCT SEARCH
# =========================================================
def query_products(user_query: str, brand_norm: Optional[str]) -> List[Dict[str, Any]]:
    qvec = embed(user_query)

    flt = None
    if brand_norm:
        flt = Filter(must=[FieldCondition(key=BRAND_NORM_KEY, match=MatchValue(value=brand_norm))])

    res = qdrant.query_points(
        collection_name=PRODUCTS_COL,
        query=qvec,
        query_filter=flt,
        limit=TOP_K_PRODUCTS,
        with_payload=True,
        with_vectors=False,
    )

    products = []
    for pt in res.points:
        p = pt.payload or {}
        products.append(
            {
                "product_id": p.get(PRODUCT_ID_KEY),
                "brand": p.get(BRAND_DISPLAY_KEY),
                "name": p.get(NAME_KEY),
                "category": p.get(CATEGORY_KEY),
                "subcategory": p.get(SUBCATEGORY_KEY),
                "price": p.get(PRICE_KEY),
                "size": p.get(SIZE_KEY),
                "image_url": p.get(IMAGE_URL_KEY),
                "skin_type": p.get(SKIN_TYPE_KEY),
                "skin_concerns": p.get(SKIN_CONCERN_KEY),
                "description": p.get(DESCRIPTION_KEY),
                "important_ingredients": p.get(IMPORTANT_INGREDIENTS_KEY),
                "how_to_use": p.get(HOW_TO_USE_KEY),
                "score": float(pt.score),
            }
        )
    return products


# =========================================================
# REVIEW SEARCH (JOIN KEY)
# =========================================================
def query_reviews_for_product(qvec: List[float], product_id: str, limit: int) -> List[Dict[str, Any]]:
    if not product_id:
        return []

    flt = Filter(must=[FieldCondition(key=REVIEW_JOIN_KEY, match=MatchValue(value=product_id))])

    res = qdrant.query_points(
        collection_name=REVIEWS_COL,
        query=qvec,
        query_filter=flt,
        limit=limit,
        with_payload=True,
        with_vectors=False,
    )

    reviews = []
    for pt in res.points:
        p = pt.payload or {}
        reviews.append(
            {
                "review_id": p.get(REVIEW_ID_KEY),
                "review_text": p.get(REVIEW_TEXT_KEY),
                "rating": p.get(REVIEW_RATING_KEY),
                "review_date": p.get(REVIEW_DATE_KEY),
                "score": float(pt.score),
            }
        )
    return reviews


def avg_review_score(reviews: List[Dict[str, Any]]) -> float:
    return sum(r.get("score", 0.0) for r in reviews) / len(reviews) if reviews else 0.0


# =========================================================
# ON-DEMAND REVIEW INSIGHTS (summary + pros/cons)
# Uses ONLY already-retrieved reviews passed from UI.
# =========================================================
def make_review_insights(product: Dict[str, Any], reviews: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Generates review_summary + pros_summary + cons_summary
    from ALREADY retrieved reviews.
    NO bullets. NO other logic changes.
    """
    if not reviews:
        return {
            "review_summary": "No reviews available.",
            "pros_summary": "Not available.",
            "cons_summary": "Not available.",
        }

    # Keep prompt small and fast
    N = min(12, len(reviews))
    lines = []
    for r in reviews[:N]:
        txt = (r.get("review_text") or "").replace("\n", " ").strip()[:350]
        rating = r.get("rating")
        rating_str = f"{rating}" if rating is not None else "?"
        lines.append(f"- (rating={rating_str}) {txt}")

    product_title = f"{product.get('brand','')} — {product.get('name','')}".strip(" —")

    prompt = f"""
Summarize customer reviews for the skincare product below, using ONLY the reviews provided.

Product: {product_title}

Reviews:
{chr(10).join(lines)}

Return JSON with:
- review_summary: 2–3 sentences
- pros_summary: 1–2 sentences summarizing the main positives
- cons_summary: 1–2 sentences summarizing the main negatives

Rules:
- Only claim what is supported by the reviews.
- If any review mentions a negative experience, always include it in cons_summary.
- State the negative plainly and neutrally, without mentioning how often it appears.
- If multiple different negatives appear, summarize them briefly in one sentence.
- If no negative experience is mentioned at all, write: "No notable drawbacks were reported."
"""

    # ---------- Responses API ----------
    try:
        resp = client.responses.create(
            model=REVIEW_SUMMARY_MODEL,
            input=prompt,
            response_format={"type": "json_object"},
        )

        text = getattr(resp, "output_text", None)
        if not text:
            text = str(resp)

        data = json.loads(text)

        return {
            "review_summary": str(data.get("review_summary", "")).strip(),
            "pros_summary": str(data.get("pros_summary", "")).strip(),
            "cons_summary": str(data.get("cons_summary", "")).strip() or "Not commonly mentioned.",
        }

    # ---------- Chat Completions fallback ----------
    except Exception:
        try:
            resp = client.chat.completions.create(
                model=REVIEW_SUMMARY_MODEL,
                messages=[
                    {"role": "system", "content": "You output ONLY valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
            )

            data = json.loads(resp.choices[0].message.content)

            return {
                "review_summary": str(data.get("review_summary", "")).strip(),
                "pros_summary": str(data.get("pros_summary", "")).strip(),
                "cons_summary": str(data.get("cons_summary", "")).strip() or "Not commonly mentioned.",
            }

        except Exception:
            return {
                "review_summary": "Could not generate review insights.",
                "pros_summary": "",
                "cons_summary": "",
            }


# =========================================================
# TWO-STAGE RANKING
# Stage B stores all_reviews so UI can generate insights later.
# =========================================================
def rank_products_two_stage(user_query: str, products: List[Dict[str, Any]]) -> Dict[str, Any]:
    qvec = embed(user_query)
    candidates = products[:TOP_N_PRODUCTS]

    stage_a = []
    for p in candidates:
        reviews20 = query_reviews_for_product(qvec, p["product_id"], REVIEWS_STAGE_A)
        avg20 = avg_review_score(reviews20)
        final20 = 0.6 * p["score"] + 0.4 * avg20

        stage_a.append(
            {
                **p,
                "review_avg": avg20,
                "final_score": final20,
                "reviews_used": REVIEWS_STAGE_A,
            }
        )

    stage_a.sort(key=lambda x: x["final_score"], reverse=True)

    stage_b = []
    for p in stage_a[:TOP_DEEP_PRODUCTS]:
        reviews50 = query_reviews_for_product(qvec, p["product_id"], REVIEWS_STAGE_B)
        avg50 = avg_review_score(reviews50)
        final50 = 0.55 * p["score"] + 0.45 * avg50

        stage_b.append(
            {
                **p,
                "review_avg": avg50,
                "final_score": final50,
                "reviews_used": REVIEWS_STAGE_B,
                "all_reviews": reviews50,  # ✅ keep for on-demand UI summarization
            }
        )

    stage_b.sort(key=lambda x: x["final_score"], reverse=True)

    best = stage_b[0] if stage_b else (stage_a[0] if stage_a else None)

    return {
        "stage_a": stage_a,
        "stage_b": stage_b,
        "best_product": best,
    }


# =========================================================
# MAIN
# =========================================================
def recommend_top_20_products(user_query: str, query_id: int = 0) -> Dict[str, Any]:
    brand_norm = extract_brand(user_query)
    price_intent = resolve_price_intent(user_query)

    products = query_products(user_query, brand_norm)

    # fallback if brand filter gives nothing
    if brand_norm and not products:
        products = query_products(strip_brand_phrase(user_query), None)

    if not products:
        return {"message": "No products found for your query. Try removing brand/price constraints."}

    if price_intent:
        products = apply_price_filter(products, price_intent)
        if not products:
            return {"message": "There is no product within your request."}

    rank_pack = rank_products_two_stage(user_query, products)

    if not rank_pack.get("best_product"):
        return {"message": "No products could be ranked (empty results)."}

    # ✅ CSV logging (top 5 stage A + all stage B)
    try:
        log_recommendations_to_csv(
            query_id=query_id,
            query_text=user_query,
            stage_a=rank_pack["stage_a"],
            stage_b=rank_pack["stage_b"],
            extracted_brand_norm=brand_norm,
            used_brand_filter=bool(brand_norm),
            price_intent=price_intent,
        )
    except Exception as e:
        # We do NOT break recommendations if logging fails
        print(f"[WARN] CSV logging failed: {e}")

    return {
        # optional metadata (UI can show these if you want)
        "extracted_brand_norm": brand_norm,
        "used_brand_filter": bool(brand_norm),
        "price_intent": price_intent,
        "stage_a": rank_pack["stage_a"],
        "stage_b": rank_pack["stage_b"],
        "best_product": rank_pack["best_product"],
        "message": None,
    }


# =========================================================
# CLI BATCH TEST (RUN ONCE, EVALUATE LATER)
# =========================================================
if __name__ == "__main__":

    test_queries = [
        # --- Descriptive (10) ---
        "I have oily, acne-prone skin with frequent breakouts around my chin and jaw. I’m looking for a gentle cleanser that contains salicylic acid and won’t dry out my skin.",
        "My skin is very dry and sensitive, especially in winter. I need a moisturizer that helps repair the skin barrier and is free of parabens above 60$.",
        "I struggle with hyperpigmentation and dark spots after acne. I want a serum that helps brighten my skin and improve texture that contains Niacinamide.",
        "I’m in my late 20s and want to start using anti-aging products that help with fine lines but won’t irritate my skin.",
        "My skin gets oily during the day but feels tight after washing. I want a sunscreen that hydrates while controlling oil under 50$.",
        "I have rosacea-prone skin and experience frequent redness. I need a calming moisturizer with soothing ingredients like centella or allantoin.",
        "My skin feels rough and dull. I’m looking for a gentle exfoliating product with PHA suitable for sensitive skin.",
        "I have dry under-eyes and fine lines. Recommend a hydrating eye cream that improves elasticity.",
        "My pores look enlarged and my skin gets shiny quickly. I want a serum that helps minimize pores.",
        "My skin reacts easily to fragrance. I need a fragrance-free cleanser for daily use.",



        # # --- Medium (10) ---
        "Masks for oily and acne-prone skin",
        "Oil cleanser for removing makeup and sunscreen",
        "Eye serum or eye cream for dark circles and fine lines",
        "Lightweight moisturizer for combination skin from Estee Lauder",
        "Night cream for dry skin with anti-aging benefits",
        "Hydrating facial mist for dry skin",
        "Retinol serum for beginners",
        "Makeup removing balm for sensitive skin",
        "Sleeping mask for dehydrated skin",
        "Face oil for nighttime use",


        # # --- Short (10) ---
        "face sunscreen with vitamin E",
        "toner to hydrate dry skin",
        "brightening serum",
        "barrier repair cream",
        "exfoliators for oily skin",
        "fragrance-free face wash",
        "retinol night serum",
        "hydrating sleeping mask",
        "calming face cream",
        "makeup removing balm",


        # # --- Mixed / realistic (10) 
        "My skin gets very red after washing. Recommend a serum or toner to reduce redness.",
        "Need an affordable lightweight moisturizer for oily skin from clinique.",
        "I want a spot treatment for sudden acne breakouts under 50$?",
        "My skin breaks out easily but feels dry sometimes. Recommend a moisturizer that hydrates without clogging pores with Salicylic Acid.",
        "Recommend a face serum for acne and blemishes from Paulas Choice.",
        "My skin feels tight and uncomfortable after cleansing. Recommend a gentle daily cleanser.",
        "I want to start using retinol but I’m afraid of irritation. What beginner-friendly option should I use?",
        "My makeup doesn’t come off completely with regular cleanser. Recommend a balm or oil cleanser.",
        "My skin looks dull lately. I want something to restore glow without irritating my skin.",
        "I need a nourishing face oil for dry skin that I can use at night.",

        ]

    print(f"\nWill write results to: {CSV_PATH}")
    print("Running 20 queries...\n")

    for idx, q in enumerate(test_queries, 1):
        print("\n" + "=" * 100)
        print(f"QUERY {idx}: {q}")

        out = recommend_top_20_products(q, query_id=idx)

        if out.get("message"):
            print(out["message"])
            continue

        print("\nTOP 5 (Stage A):")
        for i, p in enumerate(out["stage_a"][:5], 1):
            print(f"{i}. {p.get('brand')} - {p.get('name')} | {p.get('price')}")

        print("\nTOP 3 (Stage B):")
        for i, p in enumerate(out["stage_b"], 1):
            print(f"{i}. {p.get('brand')} - {p.get('name')} | {p.get('price')}")

    print("\nDone. All results are saved!")
