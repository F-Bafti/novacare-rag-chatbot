# 🟢 NVIDIA CUDA Customer Service Chatbot

A production-ready RAG (Retrieval-Augmented Generation) chatbot built with
**LangGraph**, **FAISS** (cuVS in production), and **Ollama** — running 100%
locally and free on a single NVIDIA GPU.

Built as a learning and interview showcase project demonstrating the
**NVIDIA RAPIDS ecosystem** and modern LLM orchestration patterns.

---

## 📸 Demo

![Gradio UI](assets/gradio_ui.png)

---

## 🏗️ Architecture

This project implements a **Self-RAG** pattern using LangGraph — a stateful
graph where each node grades its own outputs before proceeding.

```
         ┌─────────────────────────────┐
         │        User Question         │
         └─────────────┬───────────────┘
                       │
                       ▼
              ┌─────────────────┐
              │  grade_question  │  Is this NVIDIA-related?
              └────────┬────────┘
                       │
           ┌───────────┴───────────┐
           │                       │
        yes (relevant)          no (off-topic)
           │                       │
           ▼                       ▼
  ┌─────────────────┐     ┌────────────────────────┐
  │    retrieve      │     │        deflect          │
  │  (FAISS/cuVS)   │     │  "I only answer NVIDIA  │
  └────────┬────────┘     │   questions"            │
           │              └────────────────────────┘
           ▼
  ┌─────────────────┐
  │  grade_documents │  Are retrieved docs useful?
  └────────┬────────┘
           │
     ┌─────┴──────┐
     │            │
   yes            no
     │            │
     ▼            ▼
  ┌──────────┐  ┌─────────┐
  │ generate │  │ deflect │
  └────┬─────┘  └─────────┘
       │
       ▼
  ┌──────────────────┐
  │  grade_generation │  Is the answer grounded?
  └────────┬─────────┘     (hallucination check)
           │
     ┌─────┴──────┐
     │            │
   yes            no (retry up to 3x)
     │            │
     ▼            ▼
   END         generate
```

---

## 🧰 Tech Stack

| Layer | Dev Machine | Production (DGX/NGC) |
|---|---|---|
| **Data ingestion** | `pandas` | `cuDF` (GPU DataFrames) |
| **Vector search** | `FAISS` | `cuVS` CAGRA (GPU ANN) |
| **Clustering** | `scikit-learn` | `cuML` KMeans |
| **Embeddings** | `nomic-embed-text` via Ollama | same |
| **LLM** | `llama3.2:3b` via Ollama | same |
| **Orchestration** | `LangGraph` | same |
| **UI** | `Gradio` | same |

> **Note on NVIDIA RAPIDS:** The code is architected so that swapping
> `pandas → cudf` and `faiss → cuVS` is a one-line change per file.
> The dev machine has a CUDA driver version mismatch (535.274 vs RAPIDS
> requirement) which blocks GPU memory allocation for cuDF and cuML.
> cuVS is imported and documented as the production vector store.

---

## 📂 Project Structure

```
nvidia-rag-chatbot/
├── data/
│   ├── corpus.csv          # 2130 cleaned NVIDIA CUDA doc chunks
│   ├── embeddings.npy      # 768-dim embeddings (2130 × 768)
│   └── faiss.index         # FAISS flat L2 index (cuVS CAGRA in prod)
├── assets/
│   └── gradio_ui.png       # Screenshot of the Gradio interface
├── ingest.py               # Data fetching, cleaning, embedding, indexing
├── state.py                # LangGraph shared state schema
├── nodes.py                # All graph node functions
├── graph.py                # LangGraph graph definition and compilation
├── app.py                  # Gradio UI
└── README.md
```

---

## 🔍 Retrieval Algorithm

### Embedding
Questions and documents are embedded using `nomic-embed-text` (768 dimensions),
running locally via Ollama. This model was chosen for its strong performance
on retrieval tasks and small footprint (~270MB).

### Indexing
Documents are indexed using **FAISS IndexFlatL2** — an exact L2 distance
search over all 2130 vectors. In production this is replaced with
**cuVS CAGRA** — NVIDIA's graph-based Approximate Nearest Neighbor (ANN)
algorithm that runs entirely on GPU and delivers 10-100x faster search.

```python
# Dev (FAISS)
index = faiss.IndexFlatL2(dim)
index.add(embeddings)
_, indices = index.search(q_embedding, k=5)

# Production (cuVS CAGRA — one-line swap)
from cuvs.neighbors import cagra
index = cagra.build(cagra.IndexParams(), embeddings)
results = cagra.search(cagra.SearchParams(), index, q_embedding, k=5)
```

### Why CAGRA?
CAGRA (CUDA Anisotropic Graph-based Retrieval Algorithm) builds a
proximity graph on the GPU and traverses it during search. It achieves
state-of-the-art recall at very high throughput — ideal for production
RAG systems serving many concurrent users.

---

## 📄 Corpus

The knowledge base is the
[NVIDIA CUDA C Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
— scraped, cleaned, and chunked into 2130 paragraphs.

### Sample documents

**Chunk 1:**
> The Graphics Processing Unit (GPU) has evolved into a highly parallel,
> multithreaded, manycore processor with tremendous computational horsepower
> and very high memory bandwidth...

**Chunk 2:**
> CUDA C++ extends C++ by allowing the programmer to define C++ functions,
> called kernels, that, when called, are executed N times in parallel by N
> different CUDA threads, as opposed to only once like regular C++ functions...

**Chunk 3:**
> The CUDA programming model assumes that the CUDA threads execute on a
> physically separate device that operates as a coprocessor to the host
> running the C++ program. This is the case, for example, when the kernels
> execute on a GPU and the rest of the C++ program executes on a CPU...

---

## 📊 Evaluation: Baseline vs RAG

We evaluated the chatbot against a pure LLM baseline (no retrieval) on 5
CUDA-specific questions using keyword overlap against known ground truth.

| # | Question | Baseline | RAG |
|---|---|---|---|
| 1 | Max threads per block | ✅ Pass | ✅ Pass |
| 2 | Warp divergence | ❌ Fail (hallucinated) | ✅ Pass |
| 3 | Unified memory | ❌ Fail (hallucinated) | ✅ Pass |
| 4 | Shared vs global memory | ✅ Pass | ❌ Fail (retrieval miss) |
| 5 | Thread hierarchy | ✅ Pass | ✅ Pass |
| | **SCORE** | **3/5** | **4/5** |

### Key Findings
- RAG eliminated hallucinations on Q2 and Q3 where the baseline LLM
  confidently gave wrong answers
- Q4 shows RAG's weakness — retrieval quality is the bottleneck, not
  the LLM. A cross-encoder reranker or sliding window chunking would fix this
- In production, cuVS CAGRA would deliver 10-100x faster retrieval than FAISS

---

## 💬 Example Interactions

**Q: What is CUDA and how does it work?**
> CUDA (Compute Unified Device Architecture) is a parallel computing platform
> and programming model developed by NVIDIA. It allows developers to harness
> the power of NVIDIA GPUs for general-purpose computing...

**Q: What are CUDA threads and blocks?**
> In CUDA, threads are the basic unit of parallel execution. They are grouped
> into blocks, and blocks are organized into a grid. Each thread has a unique
> ID within its block, and each block has a unique ID within the grid...

**Q: What is the difference between shared and global memory?**
> Global memory is accessible by all threads across all blocks and persists
> for the lifetime of the application, but has higher latency. Shared memory
> is on-chip memory shared only within a thread block — much faster than
> global memory but limited in size (typically 48KB per block)...

**Q: What is the weather today?** *(off-topic)*
> I'm sorry, I can only answer questions about NVIDIA CUDA documentation.
> Please rephrase your question or ask something related to NVIDIA products.

---

## 🚀 Setup & Run

### Prerequisites
- NVIDIA GPU (RTX 3060 or better)
- CUDA driver 535+
- Miniconda or Anaconda
- Ollama installed locally

### 1. Create environment
```bash
conda create -n nvidia-rag python=3.11 -y
conda activate nvidia-rag
```

### 2. Install dependencies
```bash
pip install langchain langchain-community langgraph langchain-ollama \
            gradio faiss-cpu requests beautifulsoup4 numpy pandas

# NVIDIA RAPIDS (requires matching CUDA driver)
pip install cudf-cu12 cuml-cu12 cuvs-cu12 \
  --extra-index-url=https://pypi.nvidia.com
```

### 3. Pull models
```bash
ollama pull nomic-embed-text
ollama pull llama3.2:3b
```

### 4. Ingest documents
```bash
python ingest.py
```

### 5. Run the chatbot
```bash
python app.py
# Open http://localhost:7860
```

---

## 🎯 NVIDIA Libraries Used

| Library | Role | Status |
|---|---|---|
| **cuVS** | GPU vector search (CAGRA ANN index) | ✅ Imported, production path documented |
| **cuDF** | GPU DataFrames for data cleaning | 📝 Production swap: `import cudf as pd` |
| **cuML** | GPU clustering (KMeans) | 📝 Production swap: `from cuml.cluster import KMeans` |

---

## 👤 Author

Built as a portfolio project for NVIDIA Solutions Architect interview.
Demonstrates end-to-end RAG architecture using the NVIDIA ecosystem.
