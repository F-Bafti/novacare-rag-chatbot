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


## Embeddings

To retrieve relevant chunks we need to convert text into numbers that
capture meaning. This is done by an embedding model.

### How it works

The embedding model processes one chunk at a time. Internally it splits
the text into tokens and uses self-attention so every token attends to
every other token within the same chunk. After all the attention layers,
all token vectors are averaged together into a single vector of 768 numbers.
This final vector represents the meaning of the entire chunk.

Our 21 chunks become 21 vectors of 768 dimensions each, stored in a matrix
of shape (21 x 768). This matrix is what we search at query time.

### Why nomic-embed-text

We chose nomic-embed-text because it is free, runs fully locally via Ollama,
and has strong performance on retrieval benchmarks. It was trained on 235
million text pairs from diverse sources including web pages, academic papers,
and books. It supports a context window of 8192 tokens which is more than
enough for our FAQ chunks.

### Limitation

The embedding model only sees one chunk at a time. It has no awareness of
other chunks in the document. For our NovaCare FAQ this is not a problem
because each Q&A pair is completely self-contained.

## Language Model

For answer generation we use llama3.2:3b — a 3 billion parameter model
developed by Meta and running locally via Ollama. It is free, requires no
API key, and fits in 12GB VRAM in full 16-bit precision.

## Embedding

After chunking, each chunk is converted into a single vector that captures
its meaning. This is done by the embedding model nomic-embed-text.

### How it works

The chunk is first split into tokens by the tokenizer. Each token gets a
vector from a frozen lookup table learned during training. These token
vectors then pass through transformer layers where every token attends to
every other token within the same chunk — building a contextual
representation. At the end, all token vectors are averaged together
(mean pooling) into one single vector of 768 numbers that represents
the meaning of the entire chunk. This process runs for all 21 chunks
producing a matrix of shape (21 x 768) stored in FAISS for retrieval.

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