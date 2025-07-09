# rag_chat_app.py

import streamlit as st
from rag_pipeline import build_rag_pipeline

# Initialize pipeline once
st.session_state.rag_pipeline = st.session_state.get('rag_pipeline', build_rag_pipeline())

# App title
st.title("💬 CrediTrust AI Assistant")
st.markdown("Ask a question based on customer complaints. The AI will generate a response and show its sources.")

# User input
user_question = st.text_input("Your question:", placeholder="e.g., What are common BNPL complaints?")

# Buttons
col1, col2 = st.columns([1, 1])
ask_clicked = col1.button("Ask")
clear_clicked = col2.button("Clear")

if clear_clicked:
    st.experimental_rerun()

if ask_clicked and user_question:
    with st.spinner("Generating response..."):
        answer, sources = st.session_state.rag_pipeline(user_question)

        # Display the answer
        st.subheader("📌 Answer")
        st.write(answer)

        # Display sources
        if sources:
            st.subheader("📚 Retrieved Sources")
            for i, chunk in enumerate(sources[:2]):
                st.markdown(f"**Source {i+1}:**")
                st.code(chunk.page_content, language='text')
        else:
            st.write("No relevant sources found.")
