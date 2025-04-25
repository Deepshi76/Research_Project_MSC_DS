# src/retrieval/build_dual_vector_store.py

"""
───────────────────────────────────────────────────────────────────────────────
build_dual_vector_store.py

This script constructs FAISS vector stores for:
  1) A global “uStore” knowledge base (upsell info),
  2) Per‑brand FAQ corpora,
  3) Per‑brand pricing corpora,

using OpenAI’s `text-embedding-ada-002` embeddings and FAISS inner‑product indices.
Each input text is chunked into overlapping windows, batch‑embedded, normalized,
and saved as a `.bin` FAISS index and a `.pkl` Python pickle of the original chunks.

USAGE:
    python -m src.retrieval.build_dual_vector_store

OUTPUTS (under Processed/):
    • faiss_ustore.bin   + .pkl
    • faiss_faq_<brand>.bin   + .pkl  (for each config/*.yaml brand)
    • faiss_price_<brand>.bin + .pkl
───────────────────────────────────────────────────────────────────────────────
"""

import os
import time
import pickle
import numpy as np
import faiss
from pathlib import Path
from dotenv import load_dotenv
import openai

# Load API credentials
load_dotenv()
openai.api_key     = os.getenv("OPENAI_API_KEY")
openai.api_base    = os.getenv("OPENAI_API_BASE",    "https://api.openai.com/v1")
openai.api_type    = os.getenv("OPENAI_API_TYPE",    "openai")
openai.api_version = os.getenv("OPENAI_API_VERSION", None)

# Embedding config
EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")
BATCH_SIZE = 10
EMBED_DIM = 1536
INDEX_TYPE = faiss.IndexFlatIP

# Path setup
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "Data"
PROCESSED_DIR = BASE_DIR / "Processed"
CONFIGS_DIR = BASE_DIR / "configs"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ─── Helpers ─────────────────────────────────────────────────────────────────

def chunk_text(text: str, max_chars: int = 1000, overlap: int = 200):
    start = 0
    length = len(text)
    while start < length:
        end = min(length, start + max_chars)
        yield text[start:end]
        start += max_chars - overlap

def embed_with_openai(chunks: list[str]) -> list[list[float]]:
    embeddings = []
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        try:
            resp = openai.Embedding.create(input=batch, model=EMBED_MODEL)
            embeddings.extend([d["embedding"] for d in resp["data"]])
            print(f"✅ Embedded batch {i//BATCH_SIZE + 1}/{(len(chunks)-1)//BATCH_SIZE + 1}")
        except Exception as e:
            print(f"❌ Error on batch {i//BATCH_SIZE + 1}: {e}")
            time.sleep(5)
            resp = openai.Embedding.create(input=batch, model=EMBED_MODEL)
            embeddings.extend([d["embedding"] for d in resp["data"]])
    return embeddings

def build_faiss_index(chunks: list[str], embeddings: list[list[float]], name: str):
    arr = np.array(embeddings, dtype=np.float32)
    faiss.normalize_L2(arr)
    index = INDEX_TYPE(EMBED_DIM)
    index.add(arr)

    faiss.write_index(index, str(PROCESSED_DIR / f"{name}.bin"))
    with open(PROCESSED_DIR / f"{name}.pkl", "wb") as f:
        pickle.dump(chunks, f)

    print(f"💾 Saved index '{name}' with {len(chunks)} chunks")

# ─── Main Routine ─────────────────────────────────────────────────────────────

def main():
    print("🔍 Starting FAISS vector store build...")

    # 1️⃣ Global uStore Index
    ustore_file = DATA_DIR / "ustore.txt"
    if ustore_file.exists():
        text = ustore_file.read_text(encoding="utf-8")
        chunks = list(chunk_text(text))
        embs = embed_with_openai(chunks)
        build_faiss_index(chunks, embs, "faiss_ustore")
    else:
        print("⚠️ ustore.txt not found. Skipping upsell context.")

    # 2️⃣ Per-brand indexes
    for cfg_file in CONFIGS_DIR.glob("*.yaml"):
        brand = cfg_file.stem.lower()
        built_any = False
        print(f"\n🔄 Processing brand: {brand}")

        # 2a) FAQ
        faq_path = DATA_DIR / "Brand_FAQ" / f"{brand.capitalize()}_QA_Upsell.csv"
        if faq_path.exists():
            faq_text = faq_path.read_text(encoding="utf-8")
            faq_chunks = list(chunk_text(faq_text))
            faq_embs = embed_with_openai(faq_chunks)
            build_faiss_index(faq_chunks, faq_embs, f"faiss_faq_{brand}")
            built_any = True
        else:
            print(f"⚠️ Missing FAQ for {brand}")

        # 2b) Pricing
        price_path = DATA_DIR / "Brand_Pricing" / f"{brand}.txt"
        if price_path.exists():
            price_text = price_path.read_text(encoding="utf-8")
            price_chunks = list(chunk_text(price_text))
            price_embs = embed_with_openai(price_chunks)
            build_faiss_index(price_chunks, price_embs, f"faiss_price_{brand}")
            built_any = True
        else:
            print(f"⚠️ Missing pricing info for {brand}")

        if not built_any:
            print(f"⛔ Skipped brand '{brand}' — no valid data found.")

if __name__ == "__main__":
    main()

    # 3️⃣ General fallback conversation index from Unilever_Info.txt
    fallback_path = DATA_DIR / "Unilever_Info.txt"
    if fallback_path.exists():
        print("\n📘 Building fallback index from Unilever_Info.txt...")
        fallback_text = fallback_path.read_text(encoding="utf-8")
        fallback_chunks = list(chunk_text(fallback_text, max_chars=600, overlap=150))  # longer windows for chat
        fallback_embs = embed_with_openai(fallback_chunks)
        build_faiss_index(fallback_chunks, fallback_embs, "faiss_unilever_fallback")
    else:
        print("⚠️ Unilever_Info.txt not found in Data/. Skipping fallback index.")

