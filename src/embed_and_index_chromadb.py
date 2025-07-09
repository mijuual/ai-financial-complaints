import os
import pickle
from tqdm import tqdm
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

def embed_and_store_chroma(
    chunk_file_path='../data/chunks_with_metadata.pkl',
    persist_directory='../chroma_db',
    collection_name='complaints',
    model_name='sentence-transformers/all-MiniLM-L6-v2',
    max_chunks=5000  # ✅ Limit chunks for testing
):
    # ✅ Load the chunked data
    if not os.path.exists(chunk_file_path):
        raise FileNotFoundError(f"Chunk file '{chunk_file_path}' not found!")

    with open(chunk_file_path, 'rb') as f:
        chunk_data = pickle.load(f)
        chunk_data = chunk_data[:max_chunks]  # 🔁 Limit for development

    print(f"✅ Loaded {len(chunk_data)} chunks.")

    # ✅ Extract text and metadata
    texts = [entry['text'] for entry in chunk_data]
    metadatas = [{'complaint_id': entry['complaint_id'], 'product': entry['product']} for entry in chunk_data]
    ids = [f"chunk_{i}" for i in range(len(texts))]

    # ✅ Initialize embedding model
    print("🔍 Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(model_name=model_name)

    # ✅ Wrap each text in a LangChain Document with metadata
    documents = [
        Document(page_content=texts[i], metadata=metadatas[i])
        for i in range(len(texts))
    ]

    # ✅ Store embeddings in ChromaDB
    print("🔗 Generating embeddings and saving to ChromaDB...")
    vectordb = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name=collection_name
    )

    vectordb.persist()
    print(f"✅ Embeddings stored and Chroma DB persisted at: {persist_directory}")

# Run when executed directly
if __name__ == "__main__":
    embed_and_store_chroma()
