# ── Baseline vs RAG Evaluation ────────────────────────────────────────────
# Outputs answers for external evaluation via Claude or GPT-4
# This is the correct approach — use a stronger model as judge
# ─────────────────────────────────────────────────────────────────────────

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from graph import app
from state import GraphState

TEST_QUESTIONS = [
    "What is the monthly premium for NovaCare Plus?",
    "What is the deductible for NovaCare Elite?",
    "How many therapy sessions does NovaCare Basic cover per year?",
    "What is the ER copay for NovaCare Plus?",
    "How many days do I have to add a newborn to my plan?",
    "What is the generic medication copay?",
    "How many states does NovaCare operate in?",
    "What is the grace period for late premium payments?",
    "How much is the NovaCare Rewards points worth in premium discount?",
    "What is the out-of-pocket maximum for NovaCare Basic?",
]

GROUND_TRUTH = [
    "NovaCare Plus monthly premium is $334",
    "NovaCare Elite deductible is $850 per year",
    "NovaCare Basic covers up to 20 therapy sessions per year",
    "NovaCare Plus ER copay is $200 per visit waived if admitted",
    "31 days to add a newborn as a dependent",
    "Generic medication copay is $12",
    "NovaCare operates in 38 states",
    "31-day grace period for late premium payments",
    "1000 points equals $10 off monthly premium up to $120 per year",
    "NovaCare Basic out-of-pocket maximum is $8,700",
]

llm = ChatOllama(model="llama3.2:3b", temperature=0)

def answer_without_rag(question: str) -> str:
    response = llm.invoke([
        SystemMessage(content="You are a helpful health insurance customer service assistant."),
        HumanMessage(content=question)
    ])
    return response.content

def answer_with_rag(question: str) -> str:
    result = app.invoke(GraphState(
        question    = question,
        documents   = [],
        generation  = None,
        is_relevant = None,
        is_grounded = None,
        retries     = 0
    ))
    return result["generation"]

def run_evaluation():
    print("=" * 70)
    print("  BASELINE vs RAG EVALUATION")
    print("  Model: llama3.2:3b | Corpus: NVIDIA CUDA Programming Guide")
    print("  Grading: external (Claude / GPT-4 as judge)")
    print("=" * 70)

    all_results = []

    for i, (question, truth) in enumerate(
        zip(TEST_QUESTIONS, GROUND_TRUTH), 1
    ):
        print(f"\n{'─' * 70}")
        print(f"Q{i}: {question}")
        print(f"Reference: {truth}")
        print(f"{'─' * 70}")

        print("\n🔴 WITHOUT RAG:")
        without = answer_without_rag(question)
        print(without)

        print("\n🟢 WITH RAG:")
        with_rag = answer_with_rag(question)
        print(with_rag)

        all_results.append({
            "q": i,
            "question": question,
            "reference": truth,
            "baseline": without,
            "rag": with_rag,
        })

        # ── Save after every question ──────────────────────────────────────
        with open("data/eval_results.txt", "w") as f:
            f.write("NVIDIA RAG CHATBOT — EVALUATION RESULTS\n")
            f.write("Judge: Claude / GPT-4 (external)\n")
            f.write("=" * 70 + "\n\n")
            f.write("INSTRUCTIONS FOR JUDGE:\n")
            f.write("For each question, grade Baseline and RAG as PASS or FAIL.\n")
            f.write("A PASS means the answer is factually correct and complete.\n")
            f.write("A FAIL means the answer is wrong, hallucinated, or says 'I don't know'.\n")
            f.write("=" * 70 + "\n\n")
            for r in all_results:
                f.write(f"Q{r['q']}: {r['question']}\n")
                f.write(f"Reference: {r['reference']}\n")
                f.write(f"{'─' * 70}\n")
                f.write(f"BASELINE ANSWER:\n{r['baseline']}\n\n")
                f.write(f"RAG ANSWER:\n{r['rag']}\n\n")
                f.write(f"BASELINE GRADE: [ PASS / FAIL ]\n")
                f.write(f"RAG GRADE:      [ PASS / FAIL ]\n")
                f.write(f"{'=' * 70}\n\n")

        print(f"💾 Saved Q{i} to data/eval_results.txt")
        print(f"{'=' * 70}")

    # ── Export clean eval file for external judge ──────────────────────────
    with open("data/eval_results.txt", "w") as f:
        f.write("NVIDIA RAG CHATBOT — EVALUATION RESULTS\n")
        f.write("Judge: Claude / GPT-4 (external)\n")
        f.write("=" * 70 + "\n\n")
        f.write("INSTRUCTIONS FOR JUDGE:\n")
        f.write("For each question, grade Baseline and RAG as PASS or FAIL.\n")
        f.write("A PASS means the answer is factually correct and complete.\n")
        f.write("A FAIL means the answer is wrong, hallucinated, or says 'I don't know'.\n")
        f.write("=" * 70 + "\n\n")

        for r in all_results:
            f.write(f"Q{r['q']}: {r['question']}\n")
            f.write(f"Reference: {r['reference']}\n")
            f.write(f"{'─' * 70}\n")
            f.write(f"BASELINE ANSWER:\n{r['baseline']}\n\n")
            f.write(f"RAG ANSWER:\n{r['rag']}\n\n")
            f.write(f"BASELINE GRADE: [ PASS / FAIL ]\n")
            f.write(f"RAG GRADE:      [ PASS / FAIL ]\n")
            f.write(f"{'=' * 70}\n\n")

    print("\n✅ Results saved to data/eval_results.txt")
    print("   Open this file and paste into Claude or GPT-4 for grading!")

if __name__ == "__main__":
    run_evaluation()