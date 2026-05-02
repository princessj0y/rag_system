from .doc_cleaner import clean_doc

def run_overlapping_chunking(raw_text, chunk_size=1000, overlap=150, is_eng = False):
    # Overlap cannot be larger than the chunk itself
    if overlap >= chunk_size:
        raise ValueError("Overlap must be strictly less than chunk_size")
        
    step_size = chunk_size - overlap
    chunks = []
    
    # Create overlapping chunks
    for i in range(0, len(raw_text), step_size):
        chunk = raw_text[i : i + chunk_size]
        chunks.append(chunk)
        
        # Stop if the end of this chunk reaches or passes the end of the document
        if i + chunk_size >= len(raw_text):
            break
        
    return chunks

# --- EXECUTION ---
if __name__ == "__main__":
    file_to_analyze = "./test/CELEX_32006L0054_EN_TXT.pdf"
    chunk_size=1000
    overlap=150

    raw_text = clean_doc(file_to_analyze)
    chunks = run_overlapping_chunking(raw_text, chunk_size=chunk_size, overlap=overlap)
            
    print(f"\n--- Analysis of {file_to_analyze} ---")
    print(f"Created {len(chunks)} chunks with size {chunk_size} and overlap {overlap}.")
    
    # Print the first two chunks to see the overlap in action
    for index, chunk in enumerate(chunks[:2], start=1):
        print(f"\nChunk {index}:")
        print(chunk)
        print("-" * 30)