# rag_pipeline.py

import os
import pickle
from typing import List

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from transformers import pipeline

# Load embedding model
def load_embedding_model(model_name="sentence-transformers/all-MiniLM-L6-v2"):
    print("\n✅ Loading embedding model...")
    return HuggingFaceEmbeddings(model_name=model_name)

def load_vector_store(embedding_model=None, collection_name="complaints"):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    persist_directory = os.path.join(base_dir, "..", "chroma_db")
    print(f"✅ Loading Chroma vector store from: {persist_directory}")
    return Chroma(persist_directory=persist_directory, embedding_function=embedding_model, collection_name=collection_name)

# Build retriever
def retrieve_relevant_chunks(query: str, vectorstore, k: int = 5):
    return vectorstore.similarity_search(query, k=k)

# Prompt template
def build_prompt_template():
    template = (
        "You are a financial analyst assistant for CrediTrust. "
        "Your task is to answer questions about customer complaints. "
        "Use the following retrieved complaint excerpts to formulate your answer. "
        "If the context doesn't contain the answer, state that you don't have enough information.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n"
        "Answer:"
    )
    return PromptTemplate(input_variables=["context", "question"], template=template)

# Load a faster, lightweight LLM for development
def load_generation_model():
    print("✅ Loading fast generation model...")
    return HuggingFacePipeline(
        pipeline=pipeline(
            "text2text-generation",  # Use "text2text-generation" for T5-style models
            model="google/flan-t5-small",  # Much faster than Mistral
            tokenizer="google/flan-t5-small",
            max_new_tokens=256,
            do_sample=False
        )
    )

def evaluate_pipeline(pipeline, questions: List[str]):
    print("\n📊 RAG Pipeline Evaluation (Markdown Table)")
    print("| Question | Generated Answer | Top Source Excerpt | Quality Score (1-5) | Comments |")
    print("|----------|------------------|---------------------|----------------------|----------|")

    for q in questions:
        result = pipeline({"query": q})
        answer = result["result"]
        sources = result["source_documents"]
        source_excerpt = sources[0].page_content[:150] + "..." if sources else "No source found."

        # Placeholder score and comment for now
        quality_score = input(f"\n🔍 Rate the answer (1–5) for:\n{q}\n→ {answer.strip()}\nScore: ")
        comment = input("Comment: ")

        print(f"| {q} | {answer.strip()} | {source_excerpt} | {quality_score} | {comment} |")


# Build the full RAG pipeline
def build_rag_pipeline():
    embedding_model = load_embedding_model()
    vectorstore = load_vector_store(embedding_model=embedding_model)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    llm = load_generation_model()
    prompt = build_prompt_template()

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )
    return qa_chain

# Run the RAG pipeline
if __name__ == "__main__":
    rag_pipeline = build_rag_pipeline()

    questions = [
        "What are the most common issues with the Buy Now Pay Later service?",
        "Why are customers dissatisfied with their savings accounts?",
        "Are there any frequent complaints about credit card fraud?",
        "What kind of delays do users report for money transfers?",
        "How do customers feel about the loan approval process?"
    ]

    evaluate_pipeline(rag_pipeline, questions)

