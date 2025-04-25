# src/utils/brand_config.py

"""
BrandConfig

Loads and validates per‑brand settings from YAML files in the project‑root `configs/` folder.
Each YAML should define:
  • brand_name         – the canonical brand key
  • synonyms           – list of trigger words for brand detection
  • brand_tone         – list of tone keywords for tone checking
  • approved_keywords  – list of product‐level keywords to boost detection
  • restricted_phrases – list of phrases disallowed by compliance rules
  • faq_path           – path to that brand’s FAQ file (for reference)
"""

import yaml
from pathlib import Path
from typing import Dict

class BrandConfig:
    # ─── PROJECT‑ROOT CONFIGS DIRECTORY ────────────────────────────────────────
    CONFIG_DIR = Path(__file__).resolve().parent.parent.parent / "configs"

    def __init__(self, brand_key: str):
        """
        Load a single configs/<brand_key>.yaml into this object.
        Raises FileNotFoundError if the file doesn't exist.
        """
        key = brand_key.lower()
        path = self.CONFIG_DIR / f"{key}.yaml"
        if not path.exists():
            raise FileNotFoundError(f"Brand config not found: {path}")

        data = yaml.safe_load(path.read_text(encoding="utf-8"))

        # Required
        # Required (but default to the brand_key if missing)
        self.brand_name: str = data.get("brand_name", brand_key)

        # Optional lists (default to empty if missing)
        self.synonyms           = [s.lower() for s in data.get("synonyms", [])]
        self.brand_tone         = data.get("brand_tone", [])
        self.approved_keywords  = data.get("approved_keywords", [])
        self.restricted_phrases = data.get("restricted_phrases", [])
        self.faq_path           = data.get("faq_path", "")

    def __repr__(self):
        return (
            f"<BrandConfig {self.brand_name!r} "
            f"synonyms={self.synonyms} "
            f"tone={self.brand_tone} "
            f"approved_keys={self.approved_keywords} "
            f"restricted={self.restricted_phrases}>"
        )

    @classmethod
    def all_configs(cls) -> Dict[str, "BrandConfig"]:
        """
        Load all YAMLs in project-root/configs/ and return a dict
        mapping brand_key → BrandConfig instance.
        """
        configs: Dict[str, BrandConfig] = {}
        for cfg_file in cls.CONFIG_DIR.glob("*.yaml"):
            key = cfg_file.stem.lower()
            try:
                configs[key] = BrandConfig(key)
            except Exception as e:
                # Skip invalid or malformed configs
                print(f"⚠️  Skipping config {cfg_file.name}: {e}")
        return configs
