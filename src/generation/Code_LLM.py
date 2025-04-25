# # 🤖 LLM RESPONSE GENERATOR (WITH FULL TRANSLATION FLOW)
# # ----------------------------------------------------------
# # 1. Translates incoming message to English
# # 2. Checks for brand violations
# # 3. Retrieves relevant context from FAISS
# # 4. Generates response using GPT-4o
# # 5. Translates response BACK to Sinhala/Tamil (if needed)
# # 6. Logs full conversation
#
# import openai
# import faiss
# import pickle
# import numpy as np
# import os
#
# from dotenv import load_dotenv
# from src.translation.translator import detect_and_translate, translate_back
# from src.validation.brand_rule_checker import is_violation
# from src.utils.logger import log_chat
#
# # ✅ Load keys from .env
# load_dotenv()
#
# openai.api_key = os.getenv("OPENAI_API_KEY")
# openai.api_base = "https://api.openai.com/v1"
# openai.api_type = "openai"
# openai.api_version = None
#
# # 1️⃣ Load FAISS index and data
# def load_vector_data():
#     with open("Processed/chunks.pkl", "rb") as f: chunks = pickle.load(f)
#     with open("Processed/embeddings.pkl", "rb") as f: embeddings = pickle.load(f)
#     index = faiss.read_index("Processed/faiss_index.bin")
#     return chunks, embeddings, index
#
# # 2️⃣ Embed query for FAISS search
# def embed_query(query):
#     resp = openai.Embedding.create(
#         input=query,
#         deployment_id="text-embedding-ada-002"  # Replace if different
#     )
#     return resp['data'][0]['embedding']
#
# # 3️⃣ Retrieve most relevant chunks
# def retrieve_context(index, query_vec, chunks):
#     D, I = index.search(np.array([query_vec], dtype=np.float32), k=5)
#     return [chunks[i] for i in I[0]]
#
# # 4️⃣ Main function triggered by UI
# def handle_query(query):
#     # 🔁 Translate incoming query to English + detect original language
#     translated_query, lang = detect_and_translate(query)
#
#     # 🔐 Check for rule violation (before LLM processing)
#     violated, rule = is_violation(translated_query)
#
#     if violated:
#         response = (
#             "At Unilever, your well-being is our top priority. "
#             "All our products meet international safety standards and are rigorously tested."
#         )
#     else:
#         # 🔎 Get relevant chunks from FAISS
#         chunks, _, index = load_vector_data()
#         vector = embed_query(translated_query)
#         context = retrieve_context(index, vector, chunks)
#
#         # 🧠 Create prompt and send to GPT
#         prompt = (
#             f"Context:\n{''.join(context)}\n\n"
#             f"User Query: {translated_query}\n"
#             f"Respond clearly and professionally in 2–3 sentences."
#         )
#
#         chat = openai.ChatCompletion.create(
#             model="gpt-4o",
#             messages=[
#                 {"role": "system", "content": "You are a helpful brand assistant."},
#                 {"role": "user", "content": prompt}
#             ]
#         )
#
#         response = chat['choices'][0]['message']['content'].strip()
#
#     # 🔁 Translate back to original language if Sinhala or Tamil
#     final_response = translate_back(response, lang)
#
#     # 🧾 Log everything
#     log_chat(query, lang, violated, final_response)
#
#     return final_response
