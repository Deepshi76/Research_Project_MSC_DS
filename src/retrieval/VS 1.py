# # 🧠 FAISS VECTOR STORE BUILDER
# # ------------------------------------------------
# # 1. Reads fair.txt
# # 2. Chunks into small parts
# # 3. Uses GPT embedding model to vectorize each chunk
# # 4. Stores: FAISS index, chunks, embeddings
#
# import os, faiss, pickle, time
# import numpy as np
# import openai
# from dotenv import load_dotenv
#
# # ✅ Load environment variables from .env
# load_dotenv()
#
# openai.api_key = os.getenv("OPENAI_API_KEY")
# openai.api_base = os.getenv("OPENAI_API_BASE")
# openai.api_type = "openai"
# openai.api_version = os.getenv("OPENAI_API_VERSION")  # Only needed for Azure
#
# # 🔧 Set this to your OpenAI/Azure deployment name
# DEPLOYMENT_ID = "text-embedding-ada-002"  # change if using Azure custom name
#
# # 📁 File paths
# TEXT_PATH = "Data/fair.txt"
# FAISS_PATH = "Processed/faiss_index.bin"
# CHUNKS_PATH = "Processed/chunks.pkl"
# EMBED_PATH = "Processed/embeddings.pkl"
#
# # 1️⃣ Read text
# def read_text(path):
#     with open(path, 'r', encoding='utf-8') as f:
#         return f.read()
#
# # 2️⃣ Chunk text into manageable pieces
# def chunk_text(text, max_chunk=1000):
#     sections = text.split('\n\n')
#     chunks = []
#     for sec in sections:
#         lines = sec.splitlines()
#         current = ""
#         for line in lines:
#             if len(current) + len(line) < max_chunk:
#                 current += line + "\n"
#             else:
#                 chunks.append(current.strip())
#                 current = line + "\n"
#         if current:
#             chunks.append(current.strip())
#     return chunks
#
# # 3️⃣ Embed chunks
# def generate_embeddings(chunks):
#     embeddings = []
#     for i in range(0, len(chunks), 10):
#         batch = chunks[i:i+10]
#         try:
#             response = openai.Embedding.create(
#                 deployment_id=DEPLOYMENT_ID,
#                 input=batch
#             )
#             batch_embeds = [e['embedding'] for e in response['data']]
#             embeddings.extend(batch_embeds)
#             print(f"✅ Embedded batch {i//10 + 1}")
#         except Exception as e:
#             print(f"❌ Error in batch {i//10 + 1}: {e}")
#             time.sleep(10)
#     return embeddings
#
# # 4️⃣ Save FAISS + pickle files
# def save_index(chunks, embeddings):
#     index = faiss.IndexFlatL2(len(embeddings[0]))
#     index.add(np.array(embeddings, dtype=np.float32))
#     faiss.write_index(index, FAISS_PATH)
#     with open(CHUNKS_PATH, "wb") as f: pickle.dump(chunks, f)
#     with open(EMBED_PATH, "wb") as f: pickle.dump(embeddings, f)
#
# # 🔁 Main runner
# def main():
#     print("📖 Reading data...")
#     text = read_text(TEXT_PATH)
#
#     print("🔧 Chunking text...")
#     chunks = chunk_text(text)
#
#     print("🧠 Embedding chunks...")
#     embeddings = generate_embeddings(chunks)
#
#     print("💾 Saving index & files...")
#     save_index(chunks, embeddings)
#
#     print("✅ Vector store ready!")
