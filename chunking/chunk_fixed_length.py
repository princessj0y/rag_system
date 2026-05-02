from .doc_cleaner import clean_doc

def run_fixed_size_chunking(raw_text, chunk_size=1000, is_eng = False):    
    # Create the chunks
    chunks = []
    for i in range(0, len(raw_text), chunk_size):
        chunks.append(raw_text[i:i + chunk_size])
    
    return chunks

# --- EXECUTION ---
if __name__ == "__main__":
    file_to_analyze = "./test/CELEX_32006L0054_EN_TXT.pdf" 
    raw_text = clean_doc(file_to_analyze)
    chunks = run_fixed_size_chunking(raw_text)
    
    # Print the chunks with labels
    print(f"\n--- Analysis of {file_to_analyze} ---")
    for index, chunk in enumerate(chunks, start=1):
        print(f"Chunk {index}:")
        print(chunk)
        print("-" * 30) # Separator line for readability