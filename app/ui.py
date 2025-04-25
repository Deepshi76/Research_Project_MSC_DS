"""
Streamlit Chat Interface

• Centers the Unilever logo, title and intro
• Captures user demographics (age bracket, gender) before chat
• Lets users optionally override brand detection
• Renders conversation history in modern chat bubbles
• Calls handle_query() on new input
"""
import sys, os
from pathlib import Path
import streamlit as st
from PIL import Image

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from src.generation.llm_generator import handle_query
from src.utils.brand_detector import load_brand_synonyms

# ─── Session init ───
if "history" not in st.session_state:
    st.session_state.history = []
if "age_bracket" not in st.session_state:
    st.session_state.age_bracket = None
if "gender" not in st.session_state:
    st.session_state.gender = None
if "brand_override" not in st.session_state:
    st.session_state.brand_override = None

# Optional memory slots
if "last_brand" not in st.session_state:
    st.session_state.last_brand = None
if "last_product" not in st.session_state:
    st.session_state.last_product = None
if "last_topic" not in st.session_state:
    st.session_state.last_topic = None

# ─── Page Setup ───
st.set_page_config(page_title="Unilever Assistant", layout="centered")

# ─── Welcome Header ───
st.image(Image.open(PROJECT_ROOT / "app" / "Unilever Logo.png"), width=120)
st.markdown("# 👋 Welcome to the Unilever Assistant")
st.markdown("I'm here to help you with product questions, prices, and recommendations.")

# ─── Collect demographics inline ───
with st.form("profile_form", clear_on_submit=False):
    col1, col2 = st.columns(2)

    with col1:
        age = st.selectbox(
            "Select your age bracket",
            ["11–18 (Teenagers)", "19–30 (Young Adults)", "31+ (Adults)"],
            index=1,
            key="age_bracket_temp"
        )

    with col2:
        gender = st.radio(
            "Select your gender",
            ["Male", "Female", "Other"],
            horizontal=True,
            key="gender_temp"
        )

    brands = sorted(load_brand_synonyms().keys())
    brand_choice = st.selectbox(
        "Optional: Override brand",
        ["Auto"] + [b.capitalize() for b in brands],
        key="brand_override_temp"
    )

    submitted = st.form_submit_button("Save & Start Chat")

if submitted:
    st.session_state.age_bracket = age
    st.session_state.gender = gender
    st.session_state.brand_override = None if brand_choice == "Auto" else brand_choice.lower()
    st.success("Profile saved. Scroll down to start chatting!")

# ─── Start Chat if profile complete ───
if st.session_state.age_bracket and st.session_state.gender:
    st.markdown("## 💬 Chat with me")
    for role, text in st.session_state.history:
        with st.chat_message(role):
            st.markdown(text)

    user_input = st.chat_input("Ask about any product...")
    if user_input:
        # 💬 Store last input for context
        st.session_state["last_user_input"] = user_input

        st.chat_message("user").markdown(user_input)
        with st.spinner("Thinking..."):
            reply = handle_query(user_input, brand_override=st.session_state.brand_override)
        st.chat_message("assistant").markdown(reply)
        st.session_state.history.append(("user", user_input))
        st.session_state.history.append(("assistant", reply))
