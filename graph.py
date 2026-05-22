from langgraph.graph import StateGraph, END
from state import GraphState
from nodes import (
    grade_question,
    retrieve,
    grade_documents,
    generate,
    grade_generation,
    deflect,
)

# ── Build the graph ───────────────────────────────────────────────────────
workflow = StateGraph(GraphState)

# ── Add nodes ─────────────────────────────────────────────────────────────
workflow.add_node("grade_question",   grade_question)
workflow.add_node("retrieve",         retrieve)
workflow.add_node("grade_documents",  grade_documents)
workflow.add_node("generate",         generate)
workflow.add_node("grade_generation", grade_generation)
workflow.add_node("deflect",          deflect)

# ── Entry point ───────────────────────────────────────────────────────────
workflow.set_entry_point("grade_question")

# ── Edge 1: is the question NVIDIA-related? ───────────────────────────────
workflow.add_conditional_edges(
    "grade_question",
    lambda state: "retrieve" if state["is_relevant"] else "deflect",
    {
        "retrieve" : "retrieve",
        "deflect"  : "deflect",
    }
)

# ── Edge 2: are the retrieved docs useful? ────────────────────────────────
workflow.add_conditional_edges(
    "grade_documents",
    lambda state: "generate" if state["is_relevant"] else "deflect",
    {
        "generate" : "generate",
        "deflect"  : "deflect",
    }
)

# ── Edge 3: is the answer grounded? (hallucination check) ─────────────────
workflow.add_conditional_edges(
    "grade_generation",
    lambda state: END if state["is_grounded"] or state["retries"] >= 3
                      else "generate",
    {
        END        : END,
        "generate" : "generate",
    }
)

# ── Linear edges ──────────────────────────────────────────────────────────
workflow.add_edge("retrieve",  "grade_documents")
workflow.add_edge("generate",  "grade_generation")
workflow.add_edge("deflect",   END)

# ── Compile ───────────────────────────────────────────────────────────────
app = workflow.compile()
print("✅ LangGraph compiled successfully")