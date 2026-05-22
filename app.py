import gradio as gr
from graph import app
from state import GraphState

# ── Run the LangGraph pipeline ─────────────────────────────────────────────
def chat(question: str) -> str:
    if not question.strip():
        return "Please ask a question."

    result = app.invoke(GraphState(
        question    = question,
        documents   = [],
        generation  = None,
        is_relevant = None,
        is_grounded = None,
        retries     = 0
    ))
    return result["generation"]

# ── Build the UI ───────────────────────────────────────────────────────────
with gr.Blocks(title="NovaCare Customer Service") as demo:

    gr.Markdown("""
    # 🏥 NovaCare Customer Service Assistant
    ### Powered by LangGraph · nomic-embed-text · llama3.2:3b · FAISS (cuVS in production)
    Ask anything about NovaCare Health Insurance plans, coverage, and policies.
    """)

    with gr.Row():
        with gr.Column(scale=4):
            question = gr.Textbox(
                placeholder="Ask a question about NovaCare...",
                label="Your Question",
                lines=3
            )
            submit = gr.Button("Ask", variant="primary")

        with gr.Column(scale=6):
            answer = gr.Textbox(
                label="Answer",
                lines=15,
                interactive=False
            )

    gr.Examples(
        examples=[
            "What is the monthly premium for NovaCare Plus?",
            "What is the deductible for NovaCare Elite?",
            "Does NovaCare cover mental health services?",
            "What is the grace period for late premium payments?",
            "How do I add a newborn to my plan?",
        ],
        inputs=question
    )

    submit.click(chat, inputs=question, outputs=answer)
    question.submit(chat, inputs=question, outputs=answer)

if __name__ == "__main__":
    demo.launch(share=False, server_port=7860)