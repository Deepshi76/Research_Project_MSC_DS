# main.py

"""
Main Pipeline Launcher

1. Verifies that FAISS indexes exist under Processed/faiss_*.bin.
2. If missing, invokes src/retrieval/build_dual_vector_store.py to build all indexes.
3. Finally, launches the Streamlit chat UI (app/ui.py).
"""

import subprocess
from pathlib import Path

# Import the "main" function from your dual‐vector builder
from src.retrieval.build_dual_vector_store import main as build_vector_store

def has_faiss_indexes(processed_dir: Path) -> bool:
    """
    Returns True if there is at least one faiss_*.bin in Processed/.
    """
    return any(processed_dir.glob("faiss_*.bin"))

def main():
    project_root = Path(__file__).resolve().parent
    processed_dir = project_root / "Processed"

    print("📦 Checking for FAISS vector indexes...")
    if not processed_dir.exists() or not has_faiss_indexes(processed_dir):
        print("⚙️  No FAISS indexes found. Building indexes now...")
        build_vector_store()
        print("✅ FAISS indexes built successfully.")
    else:
        print("✅ Existing FAISS indexes detected; skipping build.")

    print("💬 Launching Streamlit chatbot UI...")
    # On Windows you may need shell=True; on macOS/Linux this will work as-is
    subprocess.run(
        ["streamlit", "run", str(project_root / "app" / "ui.py")],
        check=True
    )

if __name__ == "__main__":
    main()
