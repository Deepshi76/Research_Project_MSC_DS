import yaml
from pathlib import Path
import random

# Load age/gender mapping categories
_CAT_PATH = Path(__file__).resolve().parent.parent.parent / "configs" / "brand_categories.yaml"
with open(_CAT_PATH, encoding="utf-8") as f:
    CATS = yaml.safe_load(f)

# Load brand range descriptions
_RANGES_PATH = Path(__file__).resolve().parent.parent.parent / "configs" / "brand_ranges.yaml"
with open(_RANGES_PATH, encoding="utf-8") as f:
    RANGES = yaml.safe_load(f)


def recommend_brands_for_user(age_bracket: str, gender: str) -> list[str]:
    """
    Recommend brands based on user age and gender.

    1. Map age to 'young' or 'mature'.
    2. Pick brand summaries from config files.
    3. Filter female-targeted brands for males.
    4. Return top 5 matches for display.
    """
    # Group logic
    group = "young" if age_bracket in ("11-18 (Teenagers)", "19-30 (Young Adults)") else "mature"

    picks = []
    for brand in CATS.get(group, []):
        summary = RANGES.get(brand)
        if not summary:
            continue
        if "women" in summary.lower() and gender == "Male":
            continue
        picks.append(f"{brand.capitalize()}: {summary}")

    # Limit to top 5 with shuffle
    random.shuffle(picks)
    picks = picks[:5]

    # Add universal Unilever brands
    for corp in CATS.get("every", []):
        picks.append(f"{corp.capitalize()}: Corporate & brand overview for all.")

    return picks