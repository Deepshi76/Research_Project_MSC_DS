# src/preprocessing/language_detection.py

import os
import logging
import pandas as pd
from transformers import pipeline
from src.translation.translate import translate_text

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Use a proper zero-shot model
language_classifier = pipeline(
    "zero-shot-classification",
    model="joeddav/xlm-roberta-large-xnli"
)

# Supported languages and mapping
CANDIDATE_LABELS = ["English", "Sinhala", "Tamil"]
LABEL_TO_CODE = {
    "English": "en",
    "Sinhala": "si",
    "Tamil": "ta"
}

def detect_language(text: str) -> str:
    try:
        text = text.strip()
        if not text or len(text.split()) < 2:
            return "unknown"
        result = language_classifier(text, candidate_labels=CANDIDATE_LABELS)
        best_label = result['labels'][0]
        return LABEL_TO_CODE.get(best_label, "unknown")
    except Exception as e:
        logger.error(f"Language detection failed: {e}")
        return "unknown"

def detect_and_translate(text: str) -> (str, str):
    lang = detect_language(text)
    if lang != "en" and lang != "unknown":
        try:
            translated = translate_text(text, src_lang=lang, tgt_lang="en")
            return lang, translated
        except Exception as e:
            logger.warning(f"Translation failed: {e}")
            return lang, text
    return lang, text

def enrich_with_language_and_translation(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("🌐 Detecting language and translating messages...")
    detected_langs = []
    translated_msgs = []

    for text in df["Inbound Message Cleaned"]:
        lang, translated = detect_and_translate(text)
        detected_langs.append(lang)
        translated_msgs.append(translated)

    df["Detected Language"] = detected_langs
    df["Inbound Message Processed"] = translated_msgs
    logger.info("✅ Language detection + translation complete.")
    return df

if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.abspath(__file__))

    input_path = os.path.join(base_dir, r"D:\Data_Science\Research\Deepshika_Rajendran_COMScDS232P-001_Research\Deepshika_Rajendran_COMScDS232P-001_Research_Code_File\Data\Processed\Cleaned_inbound_outbound_dataset.csv")
    output_path = os.path.join(base_dir, r"D:\Data_Science\Research\Deepshika_Rajendran_COMScDS232P-001_Research\Deepshika_Rajendran_COMScDS232P-001_Research_Code_File\Data\Processed\Enriched_inbound_outbound_dataset.xlsx")

    logger.info(f"📥 Loading cleaned data from: {input_path}")
    df_cleaned = pd.read_csv(input_path)

    df_enriched = enrich_with_language_and_translation(df_cleaned)

    logger.info(f"💾 Saving enriched file to: {output_path}")
    df_enriched.to_excel(output_path, index=False)
    logger.info("🎯 All done! Language and translation columns added.")
