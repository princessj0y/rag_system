import pandas # fixes pydantic segfault

import os
from .doc_cleaner import clean_doc
from langchain_text_splitters import RecursiveCharacterTextSplitter

def run_recursive_chunking(raw_text, chunk_size=1200, overlap=150, is_eng = False):
    
    # Initialize the Recursive Splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len,
        # The Waterfall: Paragraphs -> Line breaks -> Sentences -> Words -> Characters
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    # Create the chunks
    chunks = text_splitter.split_text(raw_text)
        
    return chunks

# --- EXECUTION ---
if __name__ == "__main__":
    # Make sure these paths match your folder structure!
    file_en = "./test/CELEX_32006L0054_EN_TXT.pdf"
    file_it = "./test/CELEX_32006L0054_IT_TXT.pdf"

    chunk_size = 1200
    overlap = 150

    # Run the English file first
    print(f"\n{'='*50}")
    print(f"Processing: {os.path.basename(file_en)}")
    print(f"{'='*50}")

    raw_text = clean_doc(file_en)
    chunks_en = run_recursive_chunking(raw_text, chunk_size=chunk_size, overlap=overlap)
    
    print(f"Created {len(chunks_en)} chunks.")
    print(f"Target Max Size: {chunk_size} chars | Safety Overlap: {overlap} chars\n")
    
    # Print the first 3 chunks to evaluate the boundaries
    for index, chunk in enumerate(chunks_en[:3], start=1):
        print(f"--- CHUNK {index} ({len(chunk)} characters) ---")
        print(chunk)
        print("-" * 50)

    # Run the Italian file second to compare
    print(f"\n{'='*50}")
    print(f"Processing: {os.path.basename(file_it)}")
    print(f"{'='*50}")

    raw_text = clean_doc(file_it)
    chunks_it = run_recursive_chunking(raw_text, chunk_size=chunk_size, overlap=overlap)
    
    print(f"Created {len(chunks_it)} chunks.")
    print(f"Target Max Size: {chunk_size} chars | Safety Overlap: {overlap} chars\n")
    
    # Print the first 3 chunks to evaluate the boundaries
    for index, chunk in enumerate(chunks_it[:3], start=1):
        print(f"--- CHUNK {index} ({len(chunk)} characters) ---")
        print(chunk)
        print("-" * 50)