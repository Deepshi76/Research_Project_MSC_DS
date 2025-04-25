# src/validation/tone_checker.py

"""
Tone Checker

Scores how well an LLM’s output matches a brand’s desired tone keywords.
"""

import re
from src.utils.brand_config import BrandConfig

def best_tone_score(text: str, brand: str) -> float:
    """
    Compute a [0,1] score = (# of tone keywords present) / (total tone keywords).

    Args:
      text  – the LLM’s reply
      brand – brand key (e.g. "dove")

    Returns:
      A float in [0,1], where 1.0 means all tone keywords appeared.
    """
    cfg = BrandConfig(brand)
    lowered = text.lower()
    matches = 0

    for tone_word in cfg.brand_tone:
        # match whole word
        if re.search(rf"\b{re.escape(tone_word.lower())}\b", lowered):
            matches += 1

    total = len(cfg.brand_tone) or 1
    return matches / total
