# ── NVIDIA RAPIDS NOTE ────────────────────────────────────────────────────
# Production: cuVS CAGRA for vector search (GPU-accelerated ANN)
# Dev machine: FAISS as drop-in replacement
# ─────────────────────────────────────────────────────────────────────────

import numpy as np
import pandas as pd
import faiss
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from state import GraphState

# ── Load resources once at startup ────────────────────────────────────────
df       = pd.read_csv("data/corpus.csv")
index    = faiss.read_index("data/faiss.index")
embedder = OllamaEmbeddings(model="nomic-embed-text")
llm = ChatOllama(model="llama3.2:3b", temperature=0)

# ── Node 0: Grade Question ────────────────────────────────────────────────
def grade_question(state: GraphState) -> GraphState:
    """Check if the question is relevant to NovaCare before doing anything."""
    question = state["question"]

    prompt = f"""You are a gatekeeper for NovaCare Health Insurance customer service chatbot.
Decide if the following question is related to health insurance, coverage, claims, premiums, or NovaCare services.
Reply with only YES or NO.

Question: {question}"""

    response = llm.invoke([HumanMessage(content=prompt)])
    is_relevant = "YES" in response.content.upper()
    return {**state, "is_relevant": is_relevant}


# ── Node 1: Retrieve ──────────────────────────────────────────────────────
def retrieve(state: GraphState) -> GraphState:
    """Search the FAISS index for the top-5 most relevant chunks."""
    question = state["question"]

    q_embedding = np.array(
        embedder.embed_query(question), dtype=np.float32
    ).reshape(1, -1)

    _, indices = index.search(q_embedding, k=5)
    docs = df.iloc[indices[0]]["text"].tolist()

    return {**state, "documents": docs}


# ── Node 2: Grade Documents ───────────────────────────────────────────────
def grade_documents(state: GraphState) -> GraphState:
    """Ask the LLM if the retrieved docs are relevant to the question."""
    question  = state["question"]
    documents = state["documents"]

    prompt = f"""You are a grader. Given a question and some documents,
decide if the documents are relevant.
Be generous — if ANY document is remotely relevant, reply YES.
Reply with only YES or NO.

Question: {question}
Documents: {' '.join(documents)}"""

    response = llm.invoke([HumanMessage(content=prompt)])
    is_relevant = "YES" in response.content.upper()
    return {**state, "is_relevant": is_relevant}


# ── Node 3: Generate ──────────────────────────────────────────────────────
def generate(state: GraphState) -> GraphState:
    """Generate an answer using the retrieved documents as context."""
    question  = state["question"]
    documents = state["documents"]
    retries   = state.get("retries", 0)

    system = """You are a helpful NovaCare Health Insurance customer service assistant.
Use the provided context to answer questions about NovaCare plans, coverage, claims, and policies.
Be specific and factual — quote exact numbers, dates, and policies from the context.
If the context does not contain the answer, say you don't have that information."""

    user = f"""Context:
{chr(10).join(documents)}

Question: {question}"""

    response = llm.invoke([
        SystemMessage(content=system),
        HumanMessage(content=user)
    ])

    return {**state, "generation": response.content, "retries": retries + 1}


# ── Node 4: Grade Generation ──────────────────────────────────────────────
def grade_generation(state: GraphState) -> GraphState:
    """Check if the answer is grounded in the documents."""
    documents  = state["documents"]
    generation = state["generation"]
    retries    = state.get("retries", 0)

    if retries >= 2:
        return {**state, "is_grounded": True}

    prompt = f"""You are a grader checking if an answer is grounded in the context.
Reply YES if the answer is reasonable and does not contradict the context.
Reply NO only if the answer contains clear fabrications.
Reply with only YES or NO.

Context: {' '.join(documents[:2])}
Answer: {generation}"""

    response = llm.invoke([HumanMessage(content=prompt)])
    is_grounded = "YES" in response.content.upper()
    return {**state, "is_grounded": is_grounded}


# ── Node 5: Deflect ───────────────────────────────────────────────────────
def deflect(state: GraphState) -> GraphState:
    """Handle off-topic or unanswerable questions gracefully."""
    return {**state, "generation": (
        "I'm sorry, I can only answer questions about NovaCare Health Insurance. "
        "Please ask something related to our plans, coverage, claims, or policies."
    )}