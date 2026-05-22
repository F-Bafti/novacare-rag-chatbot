# ── NovaCare Customer Service RAG ─────────────────────────────────────────
# Ingests NovaCare FAQ, chunks it, embeds it, and builds a FAISS index
# Production: cuDF + cuVS CAGRA (GPU-accelerated)
# ─────────────────────────────────────────────────────────────────────────

import os
import numpy as np
import pandas as pd                      # swap for: import cudf as pd
import faiss
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings

# ── 1. Read FAQ ───────────────────────────────────────────────────────────
print("📄 Reading NovaCare FAQ...")
with open("data/novacare_faq.txt", "r") as f:
    full_text = f.read()
print(f"   Extracted {len(full_text):,} characters")

# ── 2. Chunk by Q&A pairs ─────────────────────────────────────────────────
print("✂️  Chunking by Q&A pairs...")
chunks = []
current_chunk = []

for line in full_text.split("\n"):
    line = line.strip()
    if not line:
        continue
    # Start a new chunk at each question
    if line.startswith("Q") and line[1].isdigit() and ":" in line:
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        current_chunk = [line]
    else:
        current_chunk.append(line)

# Add the last chunk
if current_chunk:
    chunks.append(" ".join(current_chunk))

print(f"   Created {len(chunks)} chunks")

# ── 3. Clean with pandas (cuDF in production) ─────────────────────────────
print("🧹 Cleaning chunks...")
df = pd.DataFrame({"text": chunks})
df = df.drop_duplicates()
df = df.dropna()
df = df[df["text"].str.len() > 20]
df = df["text"].str.replace(r"\s+", " ", regex=True)
df = pd.DataFrame({"text": df.values})
df = df.reset_index(drop=True)
print(f"   After cleaning: {len(df)} chunks")

# ── 4. Save corpus ────────────────────────────────────────────────────────
os.makedirs("data", exist_ok=True)
df.to_csv("data/corpus.csv", index=False)
print("   Saved to data/corpus.csv")

# ── 5. Generate embeddings ────────────────────────────────────────────────
print("🔢 Generating embeddings with nomic-embed-text...")
embedder = OllamaEmbeddings(model="nomic-embed-text")
texts = df["text"].tolist()
embeddings = embedder.embed_documents(texts)
embeddings = np.array(embeddings, dtype=np.float32)
print(f"   Embeddings shape: {embeddings.shape}")

# ── 6. Build FAISS index ──────────────────────────────────────────────────
print("🗂️  Building FAISS index (cuVS CAGRA in production)...")
dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(embeddings)
print(f"   Index built with {index.ntotal} vectors")

# ── 7. Save ───────────────────────────────────────────────────────────────
np.save("data/embeddings.npy", embeddings)
faiss.write_index(index, "data/faiss.index")
print("   Saved to data/")

print("\n✅ Ingestion complete!")
print(f"   Corpus : {len(df)} chunks")
print(f"   Vectors: {embeddings.shape}")
print(f"   Index  : FAISS (cuVS CAGRA in production)")