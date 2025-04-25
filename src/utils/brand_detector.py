# src/utils/brand_detector.py

"""
Brand Detector

Loads every BrandConfig from project-root/configs/ and matches
incoming queries against each brand’s synonyms. Returns the
detected brand key (e.g. "dove"), or a fallback default.
"""

import re
from typing import Dict, Optional, List
from src.utils.brand_config import BrandConfig

def load_brand_synonyms() -> Dict[str, list[str]]:
    configs = BrandConfig.all_configs()
    return {key: cfg.synonyms for key, cfg in configs.items()}

def detect_all_brands(query: str) -> List[str]:
    text = query.lower()
    brand_synonyms = load_brand_synonyms()
    detected = []

    for brand, synonyms in brand_synonyms.items():
        for syn in synonyms:
            if re.search(rf"\b{re.escape(syn)}\b", text):
                print(f"[brand_detector] ✅ Matched brand '{brand}' via synonym '{syn}'")
                detected.append(brand)
                break  # Avoid duplicate matches

    if not detected:
        print("[brand_detector] ❌ No brand match found.")
    return detected