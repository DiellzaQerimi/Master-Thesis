import os
import uuid
import hashlib
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
import pandas as pd

from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct


# =========================================================
# PART 0 — SETTINGS
# =========================================================

load_dotenv()

# ✅ NEW collection so we don't mix old random UUID ids with new stable ids
COLLECTION_NAME = "product_reviews"

EMBED_MODEL = "text-embedding-3-small"

# Embedding batch size (safe + stable)
BATCH_SIZE = 128

# Qdrant connection
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")  # ✅ correct env var for Qdrant
QDRANT_TIMEOUT = 120

# Qdrant request safety: avoid 32MB JSON limit
UPSERT_CHUNK = 200  # 100–300 is typical

# Optional: Keep payload smaller (does NOT affect embeddings)
MAX_PAYLOAD_REVIEW_CHARS = 1200  # set to None to store full text (not recommended for huge scale)

# If you rerun tomorrow, this prevents re-embedding already stored IDs
SKIP_EXISTING_IN_QDRANT = True

# Review text used for hash/id stability (avoid hashing extreme lengths)
MAX_HASH_TEXT_CHARS = 8000

INPUT_FILES = [
    # ("Ulta_Product_Reviews.csv", "Ulta Beauty"),
    ("Sephora_Product_Reviews.csv", "Sephora"),
]


# =========================================================
# PART 1 — CLIENTS
# =========================================================

openai_client = OpenAI()
qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=QDRANT_TIMEOUT)


# =========================================================
# PART 2 — HELPERS
# =========================================================

def make_brand_product_id(brand: str, product: str) -> str:
    return f"{brand.strip().lower()} {product.strip().lower()}"

def to_iso_date(x: Any) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip()
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt).isoformat()
        except Exception:
            pass
    return s

def embed_texts(texts: List[str]) -> List[List[float]]:
    resp = openai_client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [d.embedding for d in resp.data]

def embed_texts_retry(texts: List[str], max_retries: int = 6) -> List[List[float]]:
    """
    Retries for transient network/rate-limit errors.
    Keeps your overnight run more stable.
    """
    for attempt in range(max_retries):
        try:
            return embed_texts(texts)
        except Exception as e:
            wait = min(60, 2 ** attempt)
            print(f"⚠️ Embedding error: {e} | retrying in {wait}s...")
            time.sleep(wait)
    raise RuntimeError("Embedding failed after retries.")

def get_col(df: pd.DataFrame, row: pd.Series, names: List[str], default: str = "") -> str:
    lower_map = {c.lower(): c for c in df.columns}
    for name in names:
        col = lower_map.get(name.lower())
        if col is not None:
            val = row.get(col)
            return "" if pd.isna(val) else str(val).strip()
    return default

def get_num(df: pd.DataFrame, row: pd.Series, names: List[str]) -> Optional[float]:
    s = get_col(df, row, names, default="")
    if not s:
        return None
    try:
        return float(s)
    except Exception:
        return None

def stable_review_id(
    source: str,
    brand: str,
    product: str,
    title: str,
    text: str,
    date_iso: str,
) -> str:
    """
    Deterministic ID: same review -> same ID across runs.
    If you rerun tomorrow, Qdrant will overwrite the same point instead of duplicating.
    """
    title = (title or "").strip()
    text = (text or "").strip()

    # Clamp the hashed content for performance + stability
    if len(text) > MAX_HASH_TEXT_CHARS:
        text = text[:MAX_HASH_TEXT_CHARS]

    # ✅ Include date_iso so identical text posted on different dates becomes distinct (optional but usually correct)
    raw = f"{source}|{brand}|{product}|{title}|{date_iso}|{text}"
    return hashlib.sha1(raw.encode("utf-8", errors="ignore")).hexdigest()

def qdrant_point_id_from_review_id(review_id: str) -> str:
    """
    Qdrant point IDs must be either unsigned int or UUID.
    Your review_id is a sha1 hex string, so convert it to a deterministic UUID.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_URL, review_id))


# =========================================================
# PART 3 — COLLECTION
# =========================================================

def ensure_collection(vector_size: int) -> None:
    existing = [c.name for c in qdrant.get_collections().collections]
    if COLLECTION_NAME not in existing:
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
        print(f"✅ Created collection: {COLLECTION_NAME}")
    else:
        print(f"ℹ️ Collection already exists: {COLLECTION_NAME}")


# =========================================================
# PART 4 — ROW → PAYLOAD + EMB TEXT
# =========================================================

def row_to_payload_and_text(df: pd.DataFrame, r: pd.Series, source_name: str) -> tuple[Dict[str, Any], str]:
    brand_display = get_col(df, r, ["brand"])
    product_display = get_col(df, r, ["product"])

    title = get_col(df, r, ["title"])
    review_text_raw = get_col(df, r, ["review_text", "text", "review"])

    rating = get_num(df, r, ["rating"])
    review_date = to_iso_date(get_col(df, r, ["submission_time"]))

    season = get_col(df, r, ["season_category"])
    skin_type = get_col(df, r, ["skin_type"])
    age_range = get_col(df, r, ["age"])
    

    if not brand_display or not product_display:
        raise ValueError("Missing brand/product in row. Check your CSV headers.")

    brand_product_id = make_brand_product_id(brand_display, product_display)

    # ✅ Deterministic review_id (sha1 string) stored in payload
    review_id = stable_review_id(
        source=source_name,
        brand=brand_display,
        product=product_display,
        title=title,
        text=review_text_raw,
        date_iso=review_date,
    )

    # ✅ Deterministic Qdrant point ID (UUID string)
    point_id = qdrant_point_id_from_review_id(review_id)

    # Embedding text uses FULL review text (best quality)
    emb_text = (f"{title}\n{review_text_raw}").strip() if title else (review_text_raw or "").strip()
    if not emb_text:
        return {}, ""

    # Payload can store truncated review text to keep Qdrant writes smaller
    review_text_payload = review_text_raw
    if MAX_PAYLOAD_REVIEW_CHARS is not None and review_text_payload:
        review_text_payload = review_text_payload[:MAX_PAYLOAD_REVIEW_CHARS]

    payload = {
        "review_id": review_id,       # sha1 string
        "point_id": point_id,         # UUID string used as Qdrant ID (handy for debugging)
        "source": source_name,

        "brand_display": brand_display,
        "product_display": product_display,
        "brand_product_id": brand_product_id,

        "title": title,
        "review_text": review_text_payload,

        "rating": rating,
        "review_date": review_date,
        "season": season,
        "skin_type": skin_type,
        "age_range": age_range,
    }

    return payload, emb_text


# =========================================================
# PART 5 — INGEST (RESUMABLE)
# =========================================================

def existing_ids_in_qdrant(point_ids: List[str]) -> set[str]:
    """
    Bulk check which point IDs already exist in Qdrant.
    point_ids MUST be valid Qdrant IDs (UUID strings or unsigned ints).
    """
    if not point_ids:
        return set()
    found = qdrant.retrieve(
        collection_name=COLLECTION_NAME,
        ids=point_ids,
        with_payload=False,
        with_vectors=False,
    )
    return {str(p.id) for p in found}

def ingest_file(csv_path: str, source_name: str) -> int:
    if not os.path.exists(csv_path):
        print(f"ℹ️ File not found (skipping): {csv_path}")
        return 0

    df = pd.read_csv(csv_path)
    if df.empty:
        print(f"ℹ️ Empty file (skipping): {csv_path}")
        return 0

    buffer_texts: List[str] = []
    buffer_payloads: List[Dict[str, Any]] = []
    inserted = 0
    skipped = 0

    def flush():
        nonlocal inserted, skipped, buffer_texts, buffer_payloads
        if not buffer_texts:
            return

        # ✅ Skip reviews already stored (so reruns don't re-embed)
        if SKIP_EXISTING_IN_QDRANT:
            point_ids = [p["point_id"] for p in buffer_payloads]
            found = existing_ids_in_qdrant(point_ids)

            new_texts: List[str] = []
            new_payloads: List[Dict[str, Any]] = []
            for t, p in zip(buffer_texts, buffer_payloads):
                if p["point_id"] in found:
                    skipped += 1
                else:
                    new_texts.append(t)
                    new_payloads.append(p)

            buffer_texts = new_texts
            buffer_payloads = new_payloads

            if not buffer_texts:
                return

        vectors = embed_texts_retry(buffer_texts)

        points = [
            PointStruct(id=p["point_id"], vector=v, payload=p)
            for p, v in zip(buffer_payloads, vectors)
        ]

        # ✅ Chunk upserts to avoid 32MB JSON limit
        for i in range(0, len(points), UPSERT_CHUNK):
            qdrant.upsert(collection_name=COLLECTION_NAME, points=points[i:i + UPSERT_CHUNK])

        inserted += len(points)
        buffer_texts = []
        buffer_payloads = []

    for _, r in df.iterrows():
        payload, emb_text = row_to_payload_and_text(df, r, source_name)
        if not payload or not emb_text:
            continue

        buffer_texts.append(emb_text)
        buffer_payloads.append(payload)

        if len(buffer_texts) >= BATCH_SIZE:
            flush()

    flush()
    print(f"✅ {source_name}: inserted={inserted}, skipped_existing={skipped}")
    return inserted


if __name__ == "__main__":
    # Create collection once (vector size comes from the model)
    sample_vec = embed_texts_retry(["test"])[0]
    ensure_collection(vector_size=len(sample_vec))

    total = 0
    for path, source in INPUT_FILES:
        total += ingest_file(path, source)

    print(f"\nDONE. Total new reviews embedded this run: {total}")
