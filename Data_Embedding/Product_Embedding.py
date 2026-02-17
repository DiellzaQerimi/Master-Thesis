import os
import math
import uuid
import re
import unicodedata
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, PointStruct

# Defines input dataset, Qdrant collection name, embedding configuration, and column mappings
PRODUCTS_CSV = "Full_Product_List.csv"
COLLECTION_NAME = "product_list"

EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 1536
BATCH_SIZE = 50

BRAND_COL = "brand"
PRODUCT_COL = "product"
DESC_COL = "about_the_product"
BRND_COL = "brand"
CAT_COL = "category"
SUBCAT_COL = "subcategory"

# Loads environment variables, initializes OpenAI client, and connects to local Qdrant instance
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
qdrant = QdrantClient("http://localhost:6333")

# Creates the Qdrant collection with cosine distance if it does not already exist
qdrant.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(
        size=EMBED_DIM,
        distance=Distance.COSINE
    )
)

# Loads the full product dataset used for embedding and indexing
df = pd.read_csv(PRODUCTS_CSV, low_memory=False)

# Builds a stable product identifier using brand + product fields
def build_product_id(row):
    return f"{row[BRAND_COL].strip().lower()} {row[PRODUCT_COL].strip().lower()}"

# Converts a product identifier into a deterministic UUID for Qdrant point IDs
def pid_to_uuid(pid: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, pid))

# Normalizes brand values for matching and filtering (removes accents, punctuation, casing differences)
def norm_brand(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9]+", "", s)

    return s

# Builds the text that will be embedded for each product record
def build_embedding_text(row):
    return f"""
Product Description:{row[DESC_COL]}

"""

texts = df.apply(build_embedding_text, axis=1).tolist()
product_ids = df.apply(build_product_id, axis=1).tolist()

# Generates embeddings in batches and upserts them into Qdrant with metadata payload
total_batches = math.ceil(len(texts) / BATCH_SIZE)

for batch_idx in range(total_batches):
    start = batch_idx * BATCH_SIZE
    end = min(start + BATCH_SIZE, len(texts))

    batch_texts = texts[start:end]
    batch_rows = df.iloc[start:end]
    batch_pids = product_ids[start:end]

    response = client.embeddings.create(
        model=EMBED_MODEL,
        input=batch_texts
    )

    points = []

    for i, row in enumerate(batch_rows.itertuples(index=False)):
        pid = batch_pids[i]
        vec = response.data[i].embedding

        points.append(
            PointStruct(
                id=pid_to_uuid(pid),
                vector=vec,
                payload={
                    "product_id": pid,
                    "brand_name": row.brand,
                    "norm_brand": norm_brand(row.brand),
                    "product_name": row.product,
                    "category": row.category,
                    "subcategory": row.subcategory,
                    "price": row.price,
                    "size": row.size,
                    "image_url": row.image,
                    "description": row.description,
                    "skin_type": row.skin_type,
                    "skin_concerns": row.skin_concerns,
                    "important_ingredients": row.important_ingredients,
                    "free_of": row.free_of,
                    "how_to_use": row.how_to_use
                }
            )
        )

    qdrant.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )

    print(f"Inserted batch {batch_idx + 1}/{total_batches}")

print("Product descriptions embedded successfully")
