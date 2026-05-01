from doc_cleaner import clean_doc

def run_paragraph_chunking(target_file):
    print(f"Cleaning {target_file}...")
    raw_text = clean_doc(target_file)
    
    # 1. Split the text by double new-lines (Markdown paragraph breaks)
    paragraphs = raw_text.split('\n\n')
    
    chunks = []
    
    # 2. Filter and clean the paragraphs
    for p in paragraphs:
        cleaned_p = p.strip()
        # Filter out empty lines or tiny leftover Markdown artifacts (like "---" or single letters)
        if len(cleaned_p) > 15: 
            chunks.append(cleaned_p)
            
    print(f"\n--- Paragraph Chunking Results ---")
    print(f"Created {len(chunks)} natural paragraph chunks.")
    
    # Print the first three chunks to see the natural boundaries
    for index, chunk in enumerate(chunks[:3], start=1):
        print(f"\nChunk {index}:")
        print(chunk)
        print("-" * 30)
        
    return chunks

# --- EXECUTION ---
file_to_analyze = "../test/CELEX_32006L0054_IT_TXT.pdf"
my_chunks = run_paragraph_chunking(file_to_analyze)