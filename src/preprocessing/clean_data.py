import pandas as pd
import logging
import re
from bs4 import BeautifulSoup
import os

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Patterns to detect system-generated replies
SYSTEM_PATTERNS = [
    r"thank you for taking the time to contact the consumer engagement centre",
    r"click here: %%\[.*?\]",
    r"survey",
    r"unilever consumer engagement team",
    r"%%\[.*?\]"  # CRM placeholders
]

# Generic messages to remove
GENERIC_SHORT_MESSAGES = {
    "done", "ok", "hi", "hello", "price", "how to order",
    "knorr sri lanka", "rexona sri lanka", "pond's sri lanka", "axe sri lanka"
}

def is_system_generated(text: str) -> bool:
    """Check if the text is a known system-generated message."""
    text = text.lower()
    return any(re.search(pat, text) for pat in SYSTEM_PATTERNS)

def is_meaningful_text(text: str) -> bool:
    """Check if a message is not empty, not generic, and long enough."""
    if not text or len(text.strip()) < 5:
        return False
    text = text.strip().lower()
    if text in GENERIC_SHORT_MESSAGES:
        return False
    if len(text.split()) <= 2:
        return False
    if is_system_generated(text):
        return False
    return True

def clean_text(text: str) -> str:
    """Clean text while preserving Sinhala, Tamil, and emojis."""
    if pd.isna(text):
        return ""

    text = str(text)
    text = BeautifulSoup(text, "html.parser").get_text(separator=" ")
    text = text.replace("\n", " ").replace("\r", " ")
    text = re.sub(r"%%\[[^\]]+\]", "", text)  # Remove CRM tokens
    text = re.sub(r"[^\w\s\u0D80-\u0DFF\u0B80-\u0BFF\u2600-\u27BF\U0001F300-\U0001FAFF]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean dataset by removing system/junk messages and applying proper text normalization."""
    logger.info("🧼 Starting advanced data cleaning...")

    # Step 1: Convert date columns
    date_cols = ["Date", "Created Time", "ScheduledTime", "PublishedTime"]
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
            logger.info(f"📅 Converted '{col}' to datetime.")

    # Step 2: Convert 'Associated Cases' to string
    if "Associated Cases" in df.columns:
        df["Associated Cases"] = df["Associated Cases"].fillna(0).astype(int).astype(str)
        logger.info("🔢 Converted 'Associated Cases' to string.")

    # Step 3: Drop rows where either message is missing or both are unhelpful
    if "Inbound Message" in df.columns and "Replied Post" in df.columns:
        initial_len = len(df)

        def should_keep(row):
            inbound_raw = str(row["Inbound Message"])
            reply_raw = str(row["Replied Post"])

            inbound_cleaned = clean_text(inbound_raw)
            reply_cleaned = clean_text(reply_raw)

            inbound_ok = is_meaningful_text(inbound_cleaned)
            reply_ok = reply_cleaned.strip() != ""  # ⚠ Keep only if outbound message is present

            return inbound_ok and reply_ok

        df = df[df.apply(should_keep, axis=1)].copy()
        removed = initial_len - len(df)
        logger.info(f"🗑 Removed {removed} rows with missing outbound messages or junk inbound.")

    # Step 4: Apply text cleaning to both columns
    if "Inbound Message" in df.columns:
        df["Inbound Message Cleaned"] = df["Inbound Message"].apply(clean_text)
    if "Replied Post" in df.columns:
        df["Replied Post Cleaned"] = df["Replied Post"].apply(clean_text)

    logger.info("✅ Data cleaning complete. Returning cleaned DataFrame.")
    return df

# === Optional standalone runner ===
if __name__ == '__main__':
    input_path = "D:\Data_Science\Research\Deepshika_Rajendran_COMScDS232P-001_Research\Deepshika_Rajendran_COMScDS232P-001_Research_Code_File\Data\Raw\Inbound & Outbound Dataset.xlsx"
    output_path = "D:\Data_Science\Research\Deepshika_Rajendran_COMScDS232P-001_Research\Deepshika_Rajendran_COMScDS232P-001_Research_Code_File\Data\Processed\cleaned_inbound_outbound.csv"

    df_raw = pd.read_excel(input_path)
    df_cleaned = clean_data(df_raw)
    df_cleaned.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"🎯 Cleaned dataset saved to:{output_path}")
