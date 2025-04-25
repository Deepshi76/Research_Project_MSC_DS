"""
Prompt Engineering Module

This module exposes a single function, `build_prompt`, which takes:
  • brand            – the brand key (e.g. "dove")
  • product          – detected product+price string, or None
  • sentiment        – one of "positive", "negative", or "neutral"
  • price_context    – list of retrieved pricing snippets
  • faq_context      – list of retrieved FAQ snippets
  • upsell_context   – list of retrieved uStore upsell snippets
  • user_query       – the user’s (translated) question
  • fallback_context – (optional) fallback conversational chunk(s)

It returns a single string to use as the user-message in your
OpenAI ChatCompletion, with instructions to:
  1. Adopt the appropriate tone
  2. Answer multi-part queries under conversational headings
  3. Ground entirely in the supplied contexts
  4. Never hallucinate
  5. Sound helpful, brand-aligned, and human
"""

from typing import List, Optional

def build_prompt(
    brand: str,
    product: Optional[str],
    sentiment: str,
    price_context: List[str],
    faq_context: List[str],
    upsell_context: List[str],
    user_query: str,
    fallback_context: Optional[List[str]] = None
) -> str:
    """
    Builds the final prompt for GPT based on retrieved chunks and user info.

    Args:
      brand            – brand name (e.g., "dove")
      product          – matched product line or None
      sentiment        – "positive", "neutral", or "negative"
      price_context    – top-k pricing results from FAISS
      faq_context      – top-k FAQs from FAISS
      upsell_context   – top-k upsell text from FAISS
      user_query       – the (translated) user input
      fallback_context – retrieved fallback vector context (e.g., from .txt fallback file)

    Returns:
      Final prompt string to send to GPT
    """

    # 1️⃣ Tone Selection
    if sentiment == "positive":
        tone = "Use a warm, friendly tone that reflects enthusiasm and brand positivity."
    elif sentiment == "negative":
        tone = "Use a calm, helpful tone that acknowledges the user's concern empathetically."
    else:
        tone = "Use a clear, professional, and friendly tone."

    # 2️⃣ Intent Awareness
    multi = (
        "If the user asks about multiple products or has more than one question, "
        "respond to each clearly in conversational format."
    )

    # 3️⃣ Section Builder
    def section(title: str, lines: List[str]) -> str:
        if not lines:
            return ""
        formatted = "\n".join(f"- {line}" for line in lines)
        return f"\n\n### {title}:\n{formatted}"

    # 4️⃣ Context Assembly
    context_blocks = ""
    if product:
        context_blocks += f"\n\n### Product Match:\n- {product}"
    context_blocks += section("Pricing Information", price_context)
    context_blocks += section("Frequently Asked Questions", faq_context)
    context_blocks += section("Additional Info", upsell_context)
    context_blocks += section("Conversation Info", fallback_context or [])

    # 5️⃣ Prompt Structure
    prompt = f"""You are a friendly and helpful customer assistant for the {brand.capitalize()} brand.
{tone} {multi}

Use only the information provided below. Do not invent or refer to external sources.

{context_blocks}

User's question:
\"\"\"{user_query}\"\"\"

Write a helpful, brand-aligned reply in natural conversation style. 
If the user asks about availability or price, mention https://www.ustore.lk for online shopping.
Keep it short, human, and friendly.
"""

    return prompt