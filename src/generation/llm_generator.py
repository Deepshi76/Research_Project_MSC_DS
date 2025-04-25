# src/generation/llm_generator.py

"""
LLM Response Orchestrator

This module ties together translation, compliance checks, retrieval, prompt engineering,
LLM invocation, and logging to produce risk‑aware, brand‑aligned chatbot replies.

Workflow in handle_query():
 1. Translate inco ming text to English (detect & translate)
 2. Detect sentiment, brand, and product(s)
 3. If the user asked about price but no product was detected:
    a) ask for clarification (“Which product…?”)
    b) on the follow‑up retry detection
    c) if still no product, fall back to brand recommendations using age & gender
 4. Enforce brand compliance rules (pre‑RAG)
 5. Embed the query once with OpenAI embeddings
 6. Retrieve top‑k pricing, FAQ, and uStore upsell chunks from FAISS
 7. Build a single, constrained prompt via prompt_engineering
 8. Call GPT‑4o to generate a raw reply
 9. Post‑process: check tone and hallucination
10. Append an upsell link on price questions
11. Translate back to the original language (if needed) and log everything
"""

import os
import pickle
import numpy as np
from langchain.vectorstores import FAISS
import re
from pathlib import Path
from dotenv import load_dotenv
import openai
import streamlit as st
from collections import defaultdict

from src.translation.translator import detect_and_translate, translate_back
from src.validation.brand_rule_checker import is_violation
from src.utils.brand_detector import detect_all_brands
from src.utils.product_detector import detect_products_for_brand, embed_texts as embed_query
from src.utils.sentiment import detect_sentiment
from src.utils.prompt_engineering import build_prompt
from src.validation.tone_checker import best_tone_score
from src.validation.hallucination_checker import check_hallucination
from src.utils.logger import log_chat
from src.utils.brand_recommender import recommend_brands_for_user
from src.validation.response_scorer import compute_bleu, compute_rouge, compute_f1

# ─── Load API & Paths ───
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
BASE_DIR = Path(__file__).resolve().parent.parent.parent
PROCESSED = BASE_DIR / "Processed"
PROCESSED.mkdir(exist_ok=True, parents=True)

def retrieve_chunks(index_path: Path, pkl_path: Path, query_vec: np.ndarray, k: int = 3):
    if not index_path.exists() or not pkl_path.exists():
        return [], 0.0
    index = faiss.read_index(str(index_path))
    chunks = pickle.loads(pkl_path.read_bytes())
    faiss.normalize_L2(query_vec)
    sims, idxs = index.search(query_vec, k)
    hits = [chunks[i] for i in idxs[0] if i < len(chunks)]
    avg_sim = float(np.mean(sims[0])) if sims.size else 0.0
    return hits, avg_sim

def handle_query(user_query: str, brand_override: str | None = None) -> str:
    translated, lang = detect_and_translate(user_query)
    sentiment = detect_sentiment(translated)

    # Memory handling
    vague_keywords = ["how", "is it", "can i", "does it", "should i", "daily", "available", "price"]
    is_vague = any(k in translated.lower() for k in vague_keywords)
    fallback_to_last = is_vague and st.session_state.get("last_brand") and st.session_state.get("last_product")

    if fallback_to_last:
        brands = [st.session_state["last_brand"]]
        all_products = {st.session_state["last_brand"]: [st.session_state["last_product"]]}
    else:
        brands = [brand_override] if brand_override else detect_all_brands(translated)
        if not brands and st.session_state.get("last_brand"):
            brands = [st.session_state["last_brand"]]

        all_products = {}
        for brand in brands:
            products = detect_products_for_brand(translated, brand)
            if products:
                all_products[brand] = products

    if all_products:
        for brand, prod_list in all_products.items():
            st.session_state["last_brand"] = brand
            st.session_state["last_product"] = prod_list[0]
            st.session_state["last_topic"] = translated

    if not all_products and not fallback_to_last:
        if not st.session_state.get("awaiting_product", False):
            st.session_state["awaiting_product"] = True
            return translate_back("Could you tell me which product you're referring to?", lang)

        st.session_state["awaiting_product"] = False
        age = st.session_state.get("age_bracket")
        gender = st.session_state.get("gender")
        recs = recommend_brands_for_user(age, gender)
        fallback = (
            "I couldn't find the exact product you're referring to. "
            "Based on your profile, here are some suggestions:\n\n" +
            "\n".join(f"• {r}" for r in recs[:5]) +
            "\n\n🛍️ Visit: [uStore.lk](https://www.ustore.lk)"
        )
        return translate_back(fallback, lang)

    # ✅ Smart multi-brand comparison (Signal vs Closeup)
    if len(all_products) > 1 and " or " in translated.lower():
        merged_products = []
        merged_price_ctx, merged_faq_ctx = [], []
        for brand, prods in all_products.items():
            main_prod = prods[0]
            vec = embed_query([main_prod])[0].reshape(1, -1)
            merged_products.extend(prods)
            merged_price_ctx += retrieve_chunks(PROCESSED / f"faiss_price_{brand}.bin", PROCESSED / f"faiss_price_{brand}.pkl", vec)[0]
            merged_faq_ctx += retrieve_chunks(PROCESSED / f"faiss_faq_{brand}.bin", PROCESSED / f"faiss_faq_{brand}.pkl", vec)[0]

        prompt = build_prompt(
            brand="unilever",
            product=", ".join([p.split(":")[0] for p in merged_products]),
            sentiment=sentiment,
            price_context=merged_price_ctx,
            faq_context=merged_faq_ctx,
            upsell_context=[],
            user_query=translated,
        )

        chat = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant for Unilever, making clear product comparisons."},
                {"role": "user", "content": prompt}
            ]
        )
        return translate_back(chat.choices[0].message.content.strip(), lang)

    # Standard response flow
    multi_replies = []
    for brand, prod_list in all_products.items():
        main_product = prod_list[0]
        vec = embed_query([main_product])[0].reshape(1, -1)

        price_ctx, price_sim = retrieve_chunks(PROCESSED / f"faiss_price_{brand}.bin", PROCESSED / f"faiss_price_{brand}.pkl", vec)
        faq_ctx, faq_sim     = retrieve_chunks(PROCESSED / f"faiss_faq_{brand}.bin", PROCESSED / f"faiss_faq_{brand}.pkl", vec)
        upsell_ctx, _        = retrieve_chunks(PROCESSED / "faiss_ustore.bin", PROCESSED / "faiss_ustore.pkl", vec)

        prompt = build_prompt(
            brand=brand,
            product=", ".join([p.split(":")[0] for p in prod_list]),
            sentiment=sentiment,
            price_context=price_ctx,
            faq_context=faq_ctx,
            upsell_context=upsell_ctx,
            user_query=translated,
        )

        try:
            chat = openai.ChatCompletion.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": f"You are a helpful assistant for {brand}."},
                    {"role": "user", "content": prompt}
                ]
            )
            raw = chat.choices[0].message.content.strip()
        except Exception as e:
            raw = "Sorry, I couldn't get the product info at the moment."

        tone_score = best_tone_score(raw, brand)
        halluc_score = min(price_sim, faq_sim)
        bleu = compute_bleu(raw, faq_ctx + price_ctx)
        rouge = compute_rouge(raw, faq_ctx + price_ctx)
        f1 = compute_f1(raw, faq_ctx + price_ctx)

        if brand.lower() != "unilever" and any(k in translated.lower() for k in ["price", "buy", "soap", "shampoo", "cream", "available", "how much"]):
            raw += "\n\n🛒 Visit: https://www.ustore.lk"

        multi_replies.append(raw)

        log_chat(
            user_query=user_query,
            language=lang,
            brand=brand,
            product=", ".join(prod_list),
            sentiment=sentiment,
            violation_rule=None,
            tone_score=tone_score,
            hallucination_score=halluc_score,
            bleu_score=bleu,
            rouge_score=rouge,
            f1_score=f1,
            response=raw
        )

    if multi_replies:
        return translate_back("\n\n---\n\n".join(multi_replies), lang)

    # 🔁 Fallback Vector
    query_vec = embed_query([translated])[0].reshape(1, -1)
    convo_ctx, convo_sim = retrieve_chunks(
        PROCESSED / "faiss_unilever_fallback.bin",
        PROCESSED / "faiss_unilever_fallback.pkl",
        query_vec
    )

    if convo_ctx:
        prompt = build_prompt(
            brand="unilever",
            product=None,
            sentiment=sentiment,
            price_context=[],
            faq_context=[],
            upsell_context=[],
            user_query=translated,
            fallback_context=convo_ctx
        )
        try:
            chat = openai.ChatCompletion.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a helpful, conversational brand assistant."},
                    {"role": "user", "content": prompt}
                ]
            )
            fallback_response = chat.choices[0].message.content.strip()
            return translate_back(fallback_response, lang)
        except Exception as e:
            return translate_back("Sorry, I couldn’t get the information. Please rephrase or ask again 😊", lang)

    return translate_back("Sorry, I couldn't understand that. Please ask in another way 😊", lang)