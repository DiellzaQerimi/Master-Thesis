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

# Loads environment variables and defines ingestion settings (collection, model, batching, input files)
load_dotenv()
COLLECTION_NAME = "product_reviews"
EMBED_MODEL = "text-embedding-3-small"
BATCH_SIZE = 128

# Configures Qdrant connection parameters and request sizing
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_TIMEOUT = 120

# Controls upsert chunk size and payload size limits
UPSERT_CHUNK = 200
MAX_PAYLOAD_REVIEW_CHARS = 1200
SKIP_EXISTING_IN_QDRANT = True
MAX_HASH_TEXT_CHARS = 8000

# Defines input review files and their associated source labels
INPUT_FILES = [
    ("Ulta_Product_Reviews.csv", "Ulta Beauty"),
    ("Sephora_Product_Reviews.csv", "Sephora"),
]

# Initializes OpenAI and Qdrant clients
openai_client = OpenAI()
qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=QDRANT_TIMEOUT)

# Builds a normalized brand+product identifier
def make_brand_product_id(brand: str, product: str) -> str:
    return f"{brand.strip().lower()} {product.strip().lower()}"

# Converts various date formats into ISO format when possible
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

# Generates embeddings for a list of texts
def embed_texts(texts: List[str]) -> List[List[float]]:
    resp = openai_client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [d.embedding for d in resp.data]

# Wraps embedding call with retry logic for robustness
def embed_texts_retry(texts: List[str], max_retries: int = 6) -> List[List[float]]:
    for attempt in range(max_retries):
        try:
            return embed_texts(texts)
        except Exception as e:
            wait = min(60, 2 ** attempt)
            print(f"Embedding error: {e} | retrying in {wait}s...")
            time.sleep(wait)
    raise RuntimeError("Embedding failed after retries.")

# Retrieves a column value using flexible case-insensitive matching
def get_col(df: pd.DataFrame, row: pd.Series, names: List[str], default: str = "") -> str:
    lower_map = {c.lower(): c for c in df.columns}
    for name in names:
        col = lower_map.get(name.lower())
        if col is not None:
            val = row.get(col)
            return "" if pd.isna(val) else str(val).strip()
    return default

# Retrieves a numeric column value and converts it to float
def get_num(df: pd.DataFrame, row: pd.Series, names: List[str]) -> Optional[float]:
    s = get_col(df, row, names, default="")
    if not s:
        return None
    try:
        return float(s)
    except Exception:
        return None

# Generates a deterministic review ID to avoid duplication across runs
def stable_review_id(
    source: str,
    brand: str,
    product: str,
    title: str,
    text: str,
    date_iso: str,
) -> str:
    title = (title or "").strip()
    text = (text or "").strip()

    if len(text) > MAX_HASH_TEXT_CHARS:
        text = text[:MAX_HASH_TEXT_CHARS]

    raw = f"{source}|{brand}|{product}|{title}|{date_iso}|{text}"
    return hashlib.sha1(raw.encode("utf-8", errors="ignore")).hexdigest()

# Converts a review_id hash into a deterministic UUID for Qdrant
def qdrant_point_id_from_review_id(review_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, review_id))

# Ensures the Qdrant collection exists with correct vector size and cosine distance
def ensure_collection(vector_size: int) -> None:
    existing = [c.name for c in qdrant.get_collections().collections]
    if COLLECTION_NAME not in existing:
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
        print(f"Created collection: {COLLECTION_NAME}")
    else:
        print(f"Collection already exists: {COLLECTION_NAME}")

# Converts a review row into payload and embedding text
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
        raise ValueError("Missing brand/product in row.")

    brand_product_id = make_brand_product_id(brand_display, product_display)

    review_id = stable_review_id(
        source=source_name,
        brand=brand_display,
        product=product_display,
        title=title,
        text=review_text_raw,
        date_iso=review_date,
    )

    point_id = qdrant_point_id_from_review_id(review_id)

    emb_text = (f"{title}\n{review_text_raw}").strip() if title else (review_text_raw or "").strip()
    if not emb_text:
        return {}, ""

    review_text_payload = review_text_raw
    if MAX_PAYLOAD_REVIEW_CHARS is not None and review_text_payload:
        review_text_payload = review_text_payload[:MAX_PAYLOAD_REVIEW_CHARS]

    payload = {
        "review_id": review_id,
        "point_id": point_id,
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

# Retrieves existing point IDs from Qdrant for resumable ingestion
def existing_ids_in_qdrant(point_ids: List[str]) -> set[str]:
    if not point_ids:
        return set()
    found = qdrant.retrieve(
        collection_name=COLLECTION_NAME,
        ids=point_ids,
        with_payload=False,
        with_vectors=False,
    )
    return {str(p.id) for p in found}

# Ingests a review file into Qdrant with batching and duplication protection
def ingest_file(csv_path: str, source_name: str) -> int:
    if not os.path.exists(csv_path):
        print(f"File not found (skipping): {csv_path}")
        return 0

    df = pd.read_csv(csv_path)
    if df.empty:
        print(f"Empty file (skipping): {csv_path}")
        return 0

    buffer_texts: List[str] = []
    buffer_payloads: List[Dict[str, Any]] = []
    inserted = 0
    skipped = 0

    def flush():
        nonlocal inserted, skipped, buffer_texts, buffer_payloads
        if not buffer_texts:
            return

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
    print(f"{source_name}: inserted={inserted}, skipped_existing={skipped}")
    return inserted

if __name__ == "__main__":
    sample_vec = embed_texts_retry(["test"])[0]
    ensure_collection(vector_size=len(sample_vec))

    total = 0
    for path, source in INPUT_FILES:
        total += ingest_file(path, source)

    print(f"\nDONE. Total new reviews embedded this run: {total}")
