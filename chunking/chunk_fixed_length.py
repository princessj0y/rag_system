from doc_cleaner import clean_doc

def run_fixed_size_chunking(target_file, chunk_size=1000):
    # Call the cleaner from doc_cleaner.py
    raw_text = clean_doc(target_file)
    
    # Create the chunks
    chunks = []
    for i in range(0, len(raw_text), chunk_size):
        chunks.append(raw_text[i:i + chunk_size])
    
    # Print the chunks with labels
    print(f"\n--- Analysis of {target_file} ---")
    for index, chunk in enumerate(chunks, start=1):
        print(f"Chunk {index}:")
        print(chunk)
        print("-" * 30) # Separator line for readability

# --- EXECUTION ---
# You can swap this to the Italian file easily
file_to_analyze = "../test/CELEX_32006L0054_EN_TXT.pdf" 
run_fixed_size_chunking(file_to_analyze)