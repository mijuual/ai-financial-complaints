import gradio as gr
from src.rag_pipeline import build_rag_pipeline

# Load the RAG pipeline
qa = build_rag_pipeline()

# Define the prediction function
def predict(question):
    print("\n📥 Received question:", question)

    result = qa({"query": question})
    answer = result["result"]
    print("📤 Model response:", answer)

    # Extract and format source documents
    sources = result.get("source_documents", [])
    print(f"📚 Retrieved {len(sources)} source documents")

    for i, doc in enumerate(sources[:3]):
        print(f"🔹 Source {i+1}:\n{doc.page_content[:500]}\n")

    source_texts = "\n\n---\n\n".join([doc.page_content for doc in sources[:3]]) if sources else "No sources available."

    return answer, source_texts

# Define Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("## 🔎 CrediTrust Complaint Assistant (RAG)")
    
    with gr.Row():
        with gr.Column():
            question_input = gr.Textbox(label="Your Question", placeholder="Ask about complaints, e.g., BNPL issues...")
            ask_button = gr.Button("Ask")
            clear_button = gr.Button("Clear")

        with gr.Column():
            answer_output = gr.Textbox(label="AI Answer", lines=6)
            sources_output = gr.Textbox(label="Sources (Retrieved Context)", lines=10)

    ask_button.click(fn=predict, inputs=question_input, outputs=[answer_output, sources_output])
    clear_button.click(lambda: ("", ""), None, [answer_output, sources_output])
    
# Launch the app
if __name__ == "__main__":
    demo.launch()
