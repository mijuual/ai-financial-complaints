# chunk_and_prepare_embeddings.py

import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pickle
import os

def chunk_complaints(input_csv='../data/cleaned_complaints.csv', output_file='../data/chunks_with_metadata.pkl'):
    # Load the cleaned dataset
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input CSV file '{input_csv}' not found!")

    df = pd.read_csv(input_csv)

    # Initialize LangChain text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    # List to hold chunks with metadata
    chunk_data = []

    for idx, row in df.iterrows():
        text = str(row['Consumer complaint narrative']).strip()
        if not text:
            continue

        complaint_id = row['Complaint ID']
        product = row['Product']

        split_texts = text_splitter.split_text(text)

        for chunk in split_texts:
            chunk_data.append({
                "text": chunk,
                "complaint_id": complaint_id,
                "product": product
            })

    print(f"✅ Total text chunks created: {len(chunk_data)}")

    # Save to disk
    with open(output_file, 'wb') as f:
        pickle.dump(chunk_data, f)

    print(f"✅ Chunk data saved to: {output_file}")


# Run when executed directly
if __name__ == "__main__":
    chunk_complaints()
