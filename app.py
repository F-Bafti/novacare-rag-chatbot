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
with gr.Blocks(title="NVIDIA RAG Chatbot") as demo:

    gr.Markdown("""
    # 🟢 NVIDIA CUDA Assistant
    ### Powered by LangGraph · nomic-embed-text · llama3.2:3b · FAISS (cuVS in production)
    Ask anything about NVIDIA CUDA documentation.
    """)

    with gr.Row():
        with gr.Column(scale=4):
            question = gr.Textbox(
                placeholder="Ask a question about NVIDIA CUDA...",
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
            "What is CUDA and how does it work?",
            "How does GPU memory management work in CUDA?",
            "What are CUDA threads and blocks?",
            "Explain warp execution in CUDA",
            "What is the difference between shared and global memory?",
        ],
        inputs=question
    )

    submit.click(chat, inputs=question, outputs=answer)
    question.submit(chat, inputs=question, outputs=answer)


if __name__ == "__main__":
    demo.launch(share=False, server_port=7860)