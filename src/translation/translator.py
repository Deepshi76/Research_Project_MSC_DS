# src/translation/translator.py

"""
🌐 GPT‑POWERED TRANSLATOR MODULE

1) detect_script(text) → 'si'|'ta'|'singlish'|'tanglish'|'en'
2) translate_to_english(text, lang_code)
   • Calls GPT-4o to translate non‑English scripts → English
3) translate_back(text, lang_code)
   • Only translates back for 'si' or 'ta'; returns English unchanged for others
4) detect_and_translate(text) → (english_text, lang_code)
"""
import os
from dotenv import load_dotenv
import openai

# ─── Load API Credentials ─────────────────────────────────────────────────────
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_base = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
openai.api_type = os.getenv("OPENAI_API_TYPE", "openai")
openai.api_version = os.getenv("OPENAI_API_VERSION", None)

# ─── Supported Languages ──────────────────────────────────────────────────────
_REVERSE_LANGS = {"si": "Sinhala", "ta": "Tamil"}
_ROMANIZED_TAGS = {
    "singlish": "Singlish (Sinhala in Latin script)",
    "tanglish": "Tanglish (Tamil in Latin script)"
}


# ─── Script Detection ─────────────────────────────────────────────────────────
def detect_script(text: str) -> str:
    """
    Detects the script or dialect of the input text.
    Returns: 'si' | 'ta' | 'singlish' | 'tanglish' | 'en'
    """
    # Sinhala unicode
    if any(ch in text for ch in "අආඇඈඉඊඋඌඑඒඔඕඖකඛ"):
        return "si"
    # Tamil unicode
    if any(ch in text for ch in "அஆஇஈஉஊஎஏஐஒஓஔகங"):
        return "ta"

    text_lower = text.lower()
    if any(word in text_lower for word in ["mage", "oya", "eka", "hari"]):
        return "singlish"
    if any(word in text_lower for word in ["enna", "vanga", "poda", "sapidunga"]):
        return "tanglish"

    return "en"


# ─── Translation to English ────────────────────────────────────────────────────
def translate_to_english(text: str, lang_code: str) -> str:
    if lang_code not in _REVERSE_LANGS and lang_code not in _ROMANIZED_TAGS:
        return text

    label = _REVERSE_LANGS.get(lang_code) or _ROMANIZED_TAGS.get(lang_code, "local language")
    prompt = f"Translate the following {label} sentence into English:\n\n{text}"

    try:
        resp = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a professional translator."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"⚠️ Translation to English failed ({lang_code}): {e}")
        return text  # Return original as safe fallback


# ─── Reverse Translation ──────────────────────────────────────────────────────
def translate_back(text: str, lang_code: str) -> str:
    if lang_code not in _REVERSE_LANGS:
        return text

    label = _REVERSE_LANGS[lang_code]
    prompt = f"Translate the following English sentence into {label}:\n\n{text}"

    try:
        resp = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a professional translator."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"⚠️ Reverse translation failed ({lang_code}): {e}")
        return text


# ─── Main Entrypoint ──────────────────────────────────────────────────────────
def detect_and_translate(text: str) -> tuple[str, str]:
    code = detect_script(text)
    if code == "en":
        return text, "en"
    translated = translate_to_english(text, code)
    return translated, code