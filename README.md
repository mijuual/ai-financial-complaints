#  Intelligent Complaint Analysis Assistant
### Retrieval-Augmented Generation (RAG) Chatbot for CrediTrust Financial

## 📌 Project Overview

CrediTrust Financial is a fast-growing digital finance company serving East Africa with over 500,000 users. The company receives thousands of customer complaints every month across its mobile app, email, and regulatory platforms.

This project aims to build an AI-powered internal chatbot that allows team members to ask natural-language questions about customer complaints and receive concise, evidence-backed answers.

---

## 🎯 Business Objective

Develop an internal AI assistant using Retrieval-Augmented Generation (RAG) that enables:

- Fast complaint analysis across 5 core financial products:
  - Credit Cards
  - Personal Loans
  - Buy Now, Pay Later (BNPL)
  - Savings Accounts
  - Money Transfers
- Natural-language queries from internal users (e.g., Product Managers).
- Context-rich, accurate responses sourced from actual complaint narratives.

---

## ✅ Completed Tasks – Phase 1: EDA and Preprocessing

### 🔍 Exploratory Data Analysis (EDA)
- Loaded the full CFPB Consumer Complaints dataset.
- Inspected columns, null values, and basic structure.
- Analyzed product distribution.
- Calculated and visualized narrative lengths.
- Determined presence/absence of complaint narratives.

###  Data Preprocessing
- Filtered data to include only the 5 target product categories.
- Removed records without a consumer complaint narrative.
- Cleaned text:
  - Lowercased all narratives.
  - Removed boilerplate phrases and special characters.
- Added word count column (`narrative_length`) for each complaint.

### 📊 Visualizations Generated
- Complaint count per product type.
- Histogram of narrative word counts.
- Count of missing vs. valid complaint narratives.

---

## Next Steps

###  Text Chunking, Embedding, and Vector Store Indexing
- Break long narratives into smaller, manageable text chunks.
- Convert each chunk into embeddings using transformer models.
- Store embeddings in a vector database (e.g., FAISS or Chroma).

###  Building RAG Core Logic and Evaluation
- Implement semantic retrieval to fetch top relevant complaint chunks.
- Feed results into a language model for answer generation.
- Evaluate relevance, accuracy, and clarity of outputs.

###  Creating an Interactive Chat Interface
- Build a user-friendly interface (Streamlit or Gradio).
- Support natural-language queries and filters by product/date.
- Allow users to view highlighted evidence and export summaries.

---

## 📚 Libraries Used

- **Pandas** – Data manipulation
- **NLTK** – Text tokenization
- **Matplotlib & Seaborn** – Visualizations
- **Google Colab** – Development environment
- *(Planned)* FAISS, HuggingFace Transformers, Streamlit/Gradio

---

## 🧾 Project Summary

The first phase successfully prepared a clean, filtered dataset of relevant customer complaints. This foundation enables the next phases: building semantic search capabilities, integrating with a language model, and creating a functional internal chatbot for complaint analysis.


