# src/utils/sentiment.py

"""
Sentiment Analyzer

Uses a HuggingFace Transformers pipeline (distilbert‑base‑uncased‑finetuned‑sst‑2‑english)
to classify text into positive, negative, or neutral, based on a confidence threshold.

Functions:
  detect_sentiment(text) → "positive" | "negative" | "neutral"
"""

from typing import List, Dict
from transformers import pipeline

# ─── Configuration ─────────────────────────────────────────────────────────────
_MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
_NEUTRAL_THRESH = 0.70  # if top score < this, we call it neutral
_MAX_CHARS = 512  # max characters to send to the model

# Lazy‐loaded global pipeline
_sent_pipeline = None


def _get_pipeline():
    global _sent_pipeline
    if _sent_pipeline is None:
        # return_all_scores gives both POSITIVE and NEGATIVE scores
        _sent_pipeline = pipeline(
            "sentiment-analysis",
            model=_MODEL_NAME,
            return_all_scores=True
        )
    return _sent_pipeline


def detect_sentiment(text: str) -> str:
    """
    Analyze the sentiment of `text`.

    Returns:
      - "positive" if positive score ≥ _NEUTRAL_THRESH
      - "negative" if negative score ≥ _NEUTRAL_THRESH
      - "neutral" otherwise
    """
    snippet = text[:_MAX_CHARS]
    results: List[Dict[str, float]] = _get_pipeline()(snippet)[0]
    # results example: [{'label':'NEGATIVE','score':0.12}, {'label':'POSITIVE','score':0.88}]

    # Find the label with the highest confidence
    best = max(results, key=lambda x: x["score"])
    label = best["label"].lower()
    score = best["score"]

    if score < _NEUTRAL_THRESH:
        return "neutral"
    if label in ("positive", "negative"):
        return label
    return "neutral"
