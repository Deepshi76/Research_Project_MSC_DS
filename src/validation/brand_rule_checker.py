# src/validation/brand_rule_checker.py

"""
Brand Rule Checker

Validates user queries against each brand’s restricted phrases to enforce compliance.
"""

import re
from typing import Tuple, Optional
from src.utils.brand_config import BrandConfig

def is_violation(text: str, brand: str) -> Tuple[bool, Optional[str]]:
    """
    Check if `text` contains any restricted phrase for `brand`.

    Args:
      text  – the user’s (translated) input
      brand – brand key (e.g. "dove")

    Returns:
      (violated_flag, matched_phrase_or_None)
    """
    cfg = BrandConfig(brand)
    lowered = text.lower()
    for phrase in cfg.restricted_phrases:
        # Whole‑word, case‑insensitive match
        if re.search(rf"\b{re.escape(phrase.lower())}\b", lowered):
            return True, phrase
    return False, None

