
# 🏥 NovaCare RAG Chatbot

## Overview

NovaCare Health Solutions Inc. is a fictional health insurance company
we created to build and evaluate a RAG (Retrieval-Augmented Generation)
chatbot. The company and all its policies, prices, and rules are entirely
made up — this guarantees the language model has never seen this data
during training, making it a clean testbed for measuring RAG's real impact.

The dataset contains 20 questions and answers covering NovaCare's plans,
coverage, claims, and policies. Our goal is to build a chatbot that can
accurately answer customer questions using only this document as its
knowledge base.

## Why a Fake Company?

A common mistake in RAG evaluation is using data the LLM already knows
from training. If the model can answer correctly without retrieval, you
cannot measure whether RAG is actually helping. By inventing NovaCare with
unique prices and specific policies, we guarantee the baseline LLM scores
zero without retrieval — making the value of RAG undeniable.

## The Data

The knowledge base is a single FAQ document containing 20 Q&A pairs covering
plans and premiums, deductibles and out-of-pocket maximums, coverage details
(mental health, ER, prescriptions, maternity), claims, appeals, cancellation
policies, wellness rewards, and telehealth services.

**Sample FAQ Entry**

Q1: What are NovaCare's available health insurance plans?
A: NovaCare offers three main plans: NovaCare Basic (monthly premium $187),
NovaCare Plus (monthly premium $334), and NovaCare Elite (monthly premium $521).
All plans include preventive care at no extra cost. NovaCare Elite additionally
covers dental and vision.

## Chunking Strategy

Most RAG tutorials chunk documents by character count — splitting every 500
characters with some overlap. This works for long prose but is wrong for FAQ data.

We use semantic chunking — each chunk is one complete Q&A pair. This produces
21 self-contained chunks (20 Q&A pairs + 1 header chunk). When the retriever
finds a chunk, it finds the full answer.

| Strategy | Semantic (ours) | Character-based |
|---|---|---|
| Chunk boundary | At each question | Every 500 chars |
| Each chunk | One complete Q&A | Random split |
| Context preserved | Always | Often split |
| Retrieval quality | High | Lower |

## Embedding

After chunking, each chunk is converted into a single vector that captures
its meaning. This is done by the embedding model nomic-embed-text.

### How it works

The embedding model processes one chunk at a time. The chunk is first split
into tokens by the tokenizer. Each token gets a vector from a frozen lookup
table learned during training. These token vectors then pass through
transformer layers where every token attends to every other token within
the same chunk — building a contextual representation. At the end, all
token vectors are averaged together (mean pooling) into one single vector
of 768 numbers that represents the meaning of the entire chunk. This process
runs for all 21 chunks producing a matrix of shape (21 x 768) stored in
FAISS for retrieval.

### Model details

| Property | Value |
|---|---|
| Model | nomic-embed-text |
| Parameters | 137 million |
| Output dimensions | 768 |
| Context window | 8192 tokens |
| Training data | 235 million text pairs |
| Training objective | contrastive loss |

### Training

nomic-embed-text was trained on 235 million naturally occurring text pairs
from web pages, academic papers, books, and conversation data. The training
objective is contrastive loss — similar texts are pulled closer together in
vector space and dissimilar texts are pushed apart. This is different from
LLMs which are trained to predict the next token.

### Limitation

The embedding model only sees one chunk at a time. It has no awareness of
other chunks in the document. For our NovaCare FAQ this is not a problem
because each Q&A pair is completely self-contained.

## Language Model

For answer generation we use llama3.2:3b — a 3 billion parameter model
developed by Meta and running locally via Ollama. It is free, requires no
API key, and fits in 12GB VRAM in full 16-bit precision. We also tested
llama3.1:8b which uses 4-bit quantization (8B x 4 bits / 8 = 4GB on disk)
and produces more accurate answers at the cost of speed.

Both models run entirely on the local GPU with no internet connection needed
after the initial download.

## Architecture

The chatbot is built as a stateful graph using LangGraph. Instead of a
simple linear chain, each node makes a decision and routes to the next
node based on the result. This pattern is called Self-RAG.

![LangGraph pipeline](assets/novacare_rag_graph.svg)

The graph has six nodes. Purple nodes make decisions. Teal nodes do work.
The coral node handles off-topic questions.

grade_question checks if the question is related to NovaCare. Off-topic
questions are immediately routed to deflect without any retrieval.

retrieve searches the FAISS index for the 10 most similar chunks to the
user question using vector similarity search.

grade_documents checks if the retrieved chunks are actually relevant to
the question. If not, routes to deflect.

generate sends the retrieved chunks and the question to the LLM and
produces an answer using them as context.

grade_generation checks if the answer is grounded in the retrieved chunks
and not hallucinated. If not, routes back to generate and tries again up
to 3 times.

deflect returns a polite message telling the user the question is outside
the scope of NovaCare customer service.

### Shared State

Every node reads from and writes to a shared state object that travels
through the entire graph.

| Field | Type | Description |
|---|---|---|
| question | string | the user's original question |
| documents | list | retrieved chunks from FAISS |
| generation | string | the LLM's generated answer |
| is_relevant | boolean | did retrieved docs pass grading? |
| is_grounded | boolean | is the answer grounded in the docs? |
| retries | integer | how many times we tried to generate |

### Retry Logic

The retry counter is incremented inside the generate node on every call.
The grade_generation node checks two conditions to decide where to go next.
If the answer is grounded it goes to END. If retries reaches 3 it also goes
to END to avoid infinite loops. Otherwise it routes back to generate for
another attempt.

## Retrieval

At query time the user question goes through the exact same embedding
pipeline as the chunks — tokenize, attention, mean pooling — producing
one 768-dim query vector. This vector is compared against all 21 chunk
vectors in FAISS using L2 distance. The 10 most similar chunks are
returned and passed to the LLM as context.

### Why L2 distance

L2 distance measures the straight line distance between two vectors in
768-dimensional space. Chunks that are semantically similar to the question
will have embeddings that are close together and therefore have a small L2
distance. FAISS searches all 21 vectors and returns the ones with the
smallest distance.

### Production upgrade

In production this FAISS index would be replaced with cuVS CAGRA —
NVIDIA's graph-based Approximate Nearest Neighbor algorithm that runs
entirely on GPU. CAGRA builds a proximity graph over all vectors and
traverses it during search, delivering 10-100x faster retrieval than
FAISS for large corpora. The code is architected so this is a one-line
swap.

## Evaluation

To measure the real value of RAG we compared two systems on the same
10 questions — a pure LLM baseline with no retrieval, and our full RAG
pipeline. The questions were designed to have exact answers that can only
be found in the NovaCare FAQ document.

### Why this evaluation is reliable

The FAQ was written by us specifically for this project. The exact numbers
— premiums, deductibles, copays, grace periods — do not exist anywhere on
the internet. The LLM has never seen this data during training. This
guarantees that any correct answer from the RAG system came from retrieval,
not from memorized training data. Grading was done by the author of the FAQ
who knows every correct answer with certainty.

### Results

| # | Question | Baseline | RAG |
|---|---|---|---|
| Q1 | Monthly premium for NovaCare Plus | FAIL | PASS |
| Q2 | Deductible for NovaCare Elite | FAIL | PASS |
| Q3 | Therapy sessions covered by Basic | FAIL | PASS |
| Q4 | ER copay for NovaCare Plus | FAIL | PASS |
| Q5 | Days to add a newborn to the plan | FAIL | PASS |
| Q6 | Generic medication copay | FAIL | PASS |
| Q7 | Number of states NovaCare operates in | FAIL | PASS |
| Q8 | Grace period for late payments | FAIL | PASS |
| Q9 | NovaCare Rewards points value | FAIL | FAIL |
| Q10 | Out-of-pocket maximum for Basic | FAIL | PASS |
| | **SCORE** | **0/10** | **9/10** |

### Key findings

The baseline LLM scored zero on all questions because the data is completely
unseen. On Q7 it confused NovaCare with Novant Health — a real company —
showing how LLMs hallucinate confidently when they don't know the answer.

RAG scored 9/10. The one failure (Q9) was a retrieval miss where the
grade_question node incorrectly deflected a valid NovaCare question about
the rewards program.

### Production improvements

Replacing FAISS with cuVS CAGRA would give 10-100x faster vector search
on GPU. Replacing pandas with cuDF would accelerate the ingestion pipeline.
Both are one-line swaps — the code is architected to make this trivial.

## Setup and Run

### Prerequisites

NVIDIA GPU with 8GB+ VRAM, CUDA driver 535+, Miniconda, Ollama installed locally.

### 1. Create environment

conda create -n nvidia-rag python=3.11 -y
conda activate nvidia-rag

### 2. Install dependencies

pip install langchain langchain-community langgraph langchain-ollama
pip install langchain-text-splitters gradio faiss-cpu pypdf
pip install requests beautifulsoup4 numpy pandas

### 3. Pull models

ollama pull nomic-embed-text
ollama pull llama3.2:3b

### 4. Start Ollama

ollama serve

### 5. Ingest documents

python ingest.py

### 6. Run the chatbot

python app.py

Open http://localhost:7860 in your browser.

### 7. Run evaluation

python evaluate.py
