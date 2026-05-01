from doc_cleaner import clean_doc

def run_overlapping_chunking(target_file, chunk_size=500, overlap=100):
    # Step 1: Clean the text
    raw_text = clean_doc(target_file)
    
    # Safety check: Overlap cannot be larger than the chunk itself
    if overlap >= chunk_size:
        raise ValueError("Overlap must be strictly less than chunk_size")
        
    step_size = chunk_size - overlap
    chunks = []
    
    # Step 2: Create overlapping chunks
    for i in range(0, len(raw_text), step_size):
        chunk = raw_text[i : i + chunk_size]
        chunks.append(chunk)
        
        # Stop if the end of this chunk reaches or passes the end of the document
        if i + chunk_size >= len(raw_text):
            break
            
    print(f"\n--- Analysis of {target_file} ---")
    print(f"Created {len(chunks)} chunks with size {chunk_size} and overlap {overlap}.")
    
    # Print the first two chunks to see the overlap in action
    for index, chunk in enumerate(chunks[:2], start=1):
        print(f"\nChunk {index}:")
        print(chunk)
        print("-" * 30)
        
    return chunks

# --- EXECUTION ---
file_to_analyze = "../test/CELEX_32006L0054_EN_TXT.pdf"
my_chunks = run_overlapping_chunking(file_to_analyze, chunk_size=1000, overlap=150)