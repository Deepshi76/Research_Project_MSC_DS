# src/utils/product_detector.py

"""
Product Detector

Loads raw pricing blocks from Data/Brand_Pricing/<brand>.txt, embeds them
(with OpenAI’s text-embedding-ada-002), and at inference:

  1) Embeds the user query.
  2) Computes cosine similarities against the cached block embeddings.
  3) Finds the best‐matched block, ensures it contains both 'ProductName:' and 'Cost:'.
  4) Extracts and prettifies the ProductName (splitting CamelCase) and Cost lines.
  5) Returns a string like "Dove Beauty Bathing Bar: LKR 360.00".

Block embeddings are cached in Processed/price_cache_<brand>.pkl to avoid
repeated embedding calls on every invocation.
"""
# Keep all existing imports the same
import os
import re
import pickle
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import openai

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_base = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
openai.api_type = os.getenv("OPENAI_API_TYPE", "openai")
openai.api_version = os.getenv("OPENAI_API_VERSION", None)

EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")
SIM_THRESHOLD = float(os.getenv("PRODUCT_SIM_THRESHOLD", 0.2))
FUZZY_MIN_SCORE = 0.15

BASE_DIR = Path(__file__).resolve().parent.parent.parent
PRICING_DIR = BASE_DIR / "Data" / "Brand_Pricing"
CACHE_DIR = BASE_DIR / "Processed"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

def embed_texts(texts: list[str]) -> np.ndarray:
    try:
        resp = openai.Embedding.create(input=texts, model=EMBED_MODEL)
        arr = np.array([d["embedding"] for d in resp["data"]], dtype=np.float32)
        return arr / np.linalg.norm(arr, axis=1, keepdims=True)
    except Exception as e:
        print(f"⚠️ Embedding failed: {e}")
        return np.zeros((len(texts), 1536), dtype=np.float32)

def split_camel_case(s: str) -> str:
    return re.sub(r'(?<!^)(?=[A-Z])', ' ', s)

def load_or_build_cache(brand: str) -> tuple[list[list[str]], np.ndarray]:
    cache_file = CACHE_DIR / f"price_cache_{brand}.pkl"
    if cache_file.exists():
        blocks, embs = pickle.loads(cache_file.read_bytes())
        return blocks, embs

    pricing_file = PRICING_DIR / f"{brand}.txt"
    if not pricing_file.exists():
        raise FileNotFoundError(f"No pricing file for '{brand}'")

    raw = pricing_file.read_text(encoding="utf-8")
    raw_blocks = [b.strip() for b in raw.split("\n\n") if b.strip()]
    product_blocks = [block.splitlines() for block in raw_blocks if "ProductName:" in block and "Cost:" in block]
    block_texts = [" ".join(lines) for lines in product_blocks]
    embs = embed_texts(block_texts)

    with open(cache_file, "wb") as f:
        pickle.dump((product_blocks, embs), f)

    return product_blocks, embs

def detect_products_for_brand(query: str, brand: str) -> list[str]:
    try:
        blocks, block_embs = load_or_build_cache(brand)
    except FileNotFoundError:
        return []

    q_emb = embed_texts([query])
    sims = cosine_similarity(q_emb, block_embs)[0]
    results = []

    for i, score in enumerate(sims):
        if score >= SIM_THRESHOLD or (score >= FUZZY_MIN_SCORE):
            lines = blocks[i]
            raw_name = next((l.split(":", 1)[1].strip() for l in lines if l.startswith("ProductName:")), None)
            raw_cost = next((l.split(":", 1)[1].strip() for l in lines if l.startswith("Cost:")), None)
            if raw_name and raw_cost:
                pretty_name = split_camel_case(raw_name)
                results.append(f"{pretty_name}: {raw_cost}")
    return results