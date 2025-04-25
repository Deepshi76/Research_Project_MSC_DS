# src/utils/logger.py

"""
Chat Logger

Each call to log_chat() appends one row to `Output/chat_log.csv`, recording:
  • timestamp            – ISO‑formatted datetime of the interaction
  • language             – detected language code of the user input
  • brand                – detected or overridden brand key
  • product              – “ProductName: Cost” or empty if none
  • sentiment            – “positive”/“negative”/“neutral”
  • violation_rule       – which restricted phrase triggered (empty if none)
  • tone_score           – float [0–1], how well the response matches brand tone
  • hallucination_score  – float [0–1], retrieval confidence
  • user_query           – the raw user input
  • response             – the final bot reply
"""


import csv
from pathlib import Path
from datetime import datetime

# Output path
LOG_PATH = Path("Output") / "chat_log.csv"

# Columns in log file
FIELDNAMES = [
    "timestamp",
    "language",
    "brand",
    "product",
    "sentiment",
    "violation_rule",
    "tone_score",
    "hallucination_score",
    "bleu_score",
    "rouge_score",
    "f1_score",
    "user_query",
    "response"
]

def log_chat(
    user_query: str,
    language: str,
    brand: str,
    product: str | None,
    sentiment: str,
    violation_rule: str | None,
    tone_score: float,
    hallucination_score: float,
    response: str,
    bleu_score: float = 0.0,
    rouge_score: float = 0.0,
    f1_score: float = 0.0
):
    """
    Logs one interaction to `chat_log.csv`.

    Args:
      user_query           – original input
      language             – detected language code
      brand                – brand detected or overridden
      product              – product string or empty
      sentiment            – "positive", "neutral", "negative"
      violation_rule       – if any brand rule was triggered
      tone_score           – [0–1], tone keyword match score
      hallucination_score  – [0–1], retrieval score from FAISS
      response             – the chatbot's reply
      bleu_score           – (future) BLEU metric placeholder
      rouge_score          – (future) ROUGE metric placeholder
      f1_score             – (future) F1 score placeholder
    """
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_header = not LOG_PATH.exists() or LOG_PATH.stat().st_size == 0

    with LOG_PATH.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerow({
            "timestamp": datetime.now().isoformat(),
            "language": language,
            "brand": brand,
            "product": product or "",
            "sentiment": sentiment,
            "violation_rule": violation_rule or "",
            "tone_score": f"{tone_score:.3f}",
            "hallucination_score": f"{hallucination_score:.3f}",
            "bleu_score": f"{bleu_score:.3f}",
            "rouge_score": f"{rouge_score:.3f}",
            "f1_score": f"{f1_score:.3f}",
            "user_query": user_query,
            "response": response
        })
