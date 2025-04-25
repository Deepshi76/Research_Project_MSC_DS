# src/retrieval/VS.py

import os
import pickle
import unicodedata
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer

# ─── Configuration ─────────────────────────────────────────────────────────────
EMBED_MODEL = "all-MiniLM-L6-v2"
EMBED_DIM   = 384

# Project directories
BASE_DIR    = Path(__file__).resolve().parent.parent.parent
DATA_DIR    = BASE_DIR / "Data"
PROCESSED   = DATA_DIR / "Processed"
CONFIGS_DIR = BASE_DIR / "configs"

# Initialize embedder
embedder = SentenceTransformer(EMBED_MODEL)


# ─── Helper Functions ──────────────────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = 200, overlap: int = 50):
    """
    Split `text` into word-based chunks of max `chunk_size`, overlapping by `overlap`.
    """
    words = text.split()
    for i in range(0, len(words), chunk_size - overlap):
        yield " ".join(words[i : i + chunk_size])


def build_index(chunks: list[str], index_path: Path):
    """
    Embed `chunks`, build a FAISS flat index, and save both:
      - index_path.bin  ← FAISS index
      - index_path.pkl  ← pickled list of chunks
    """
    # 1) Embed
    embeddings = embedder.encode(chunks, show_progress_bar=True)
    # 2) Normalize
    faiss.normalize_L2(embeddings)
    # 3) Create and populate FAISS index
    index = faiss.IndexFlatIP(EMBED_DIM)
    index.add(embeddings)
    # 4) Persist
    faiss.write_index(index, str(index_path.with_suffix(".bin")))
    with open(index_path.with_suffix(".pkl"), "wb") as f:
        pickle.dump(chunks, f)
    print(f" ✓ Built {index_path.name}")


def strip_accents(text: str) -> str:
    """
    Remove accents/diacritics from a Unicode string.
    """
    return "".join(
        c
        for c in unicodedata.normalize("NFKD", text)
        if not unicodedata.combining(c)
    )


def find_pricing_file(brand: str) -> Path | None:
    """
    Locate the .txt file under Data/Brand_Pricing whose stem,
    stripped of accents and lowercased, matches `brand`.
    """
    pricing_dir = DATA_DIR / "Brand_Pricing"
    for txt in pricing_dir.glob("*.txt"):
        stem = strip_accents(txt.stem).lower()
        if stem == brand.lower():
            return txt
    return None


# ─── Main Routine ──────────────────────────────────────────────────────────────

def main():
    PROCESSED.mkdir(exist_ok=True, parents=True)

    # 1) Global uStore index
    ustore_txt = DATA_DIR / "ustore.txt"
    if ustore_txt.exists():
        text = ustore_txt.read_text(encoding="utf-8")
        chunks = list(chunk_text(text))
        build_index(chunks, PROCESSED / "faiss_ustore")
    else:
        print("⚠️  ustore.txt not found; skipping uStore index.")

    # 2) Per-brand FAQ & Pricing indexes
    for cfg in CONFIGS_DIR.glob("*.yaml"):
        brand = cfg.stem.lower()
        print(f"\nBuilding indexes for brand: {brand}")

        # ── FAQ index ───────────────────────────────────────
        faq_csv = DATA_DIR / "Brand_FAQ" / f"{brand.capitalize()}_QA_Upsell.csv"
        if faq_csv.exists():
            faq_text = faq_csv.read_text(encoding="utf-8")
            faq_chunks = list(chunk_text(faq_text))
            build_index(faq_chunks, PROCESSED / f"faiss_faq_{brand}")
        else:
            print(f"⚠️  FAQ CSV for '{brand}' not found; skipping FAQ index.")

        # ── Pricing index ──────────────────────────────────
        price_file = find_pricing_file(brand)
        if price_file:
            price_text = price_file.read_text(encoding="utf-8")
            price_chunks = list(chunk_text(price_text))
            build_index(price_chunks, PROCESSED / f"faiss_price_{brand}")
        else:
            print(f"⚠️  Pricing TXT for '{brand}' not found; skipping pricing index.")


if __name__ == "__main__":
    main()
