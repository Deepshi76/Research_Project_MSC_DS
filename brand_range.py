import yaml
import csv
from pathlib import Path

# Paths
FAQ_DIR   = Path("Data/Brand_FAQ")
PRICE_DIR = Path("Data/Brand_Pricing")
OUT_PATH  = Path("configs/brand_ranges.yaml")

ranges = {}

# 1) From FAQ CSV: grab the first few distinct products/topics
for csv_file in FAQ_DIR.glob("*_QA_Upsell.csv"):
    brand = csv_file.stem.split("_")[0].lower()
    products = set()
    with open(csv_file, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Assume there's a 'question' column like "What different soaps does Dove offer?"
            # You could extract the product from the answer, or just list the FAQ topics:
            products.add(row["question"].split(" ")[-1].strip("?"))
            if len(products) >= 3:
                break
    ranges[brand] = f"Offers {', '.join(products)} and more."

# 2) From pricing text: pull block headers
for txt_file in PRICE_DIR.glob("*.txt"):
    brand = txt_file.stem.lower()
    lines = txt_file.read_text().splitlines()
    names = [l.split(":",1)[1].strip() for l in lines if l.startswith("ProductName:")]
    summary = ", ".join(names[:3])
    ranges[brand] = f"Key products include {summary}…"

# 3) Write out YAML
with open(OUT_PATH, "w", encoding="utf-8") as f:
    yaml.dump(ranges, f, sort_keys=False, allow_unicode=True)
