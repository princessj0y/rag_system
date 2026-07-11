
def run_paragraph_chunking(raw_text, is_eng = False):
    
    # Split the text by double new-lines (Markdown paragraph breaks)
    paragraphs = raw_text.split('\n\n')
    
    chunks = []
    
    # Filter and clean the paragraphs
    for p in paragraphs:
        cleaned_p = p.strip()
        # Filter out empty lines or tiny leftover Markdown artifacts (like "---" or single letters)
        if len(cleaned_p) > 15: 
            chunks.append(cleaned_p)
        
    return chunks

# --- EXECUTION ---
if __name__ == "__main__":
    from .doc_cleaner import clean_doc
    
    file_to_analyze = "./test/CELEX_32006L0054_IT_TXT.pdf"
    raw_text = clean_doc(file_to_analyze)
    chunks = run_paragraph_chunking(raw_text)
            
    print(f"\n--- Paragraph Chunking Results ---")
    print(f"Created {len(chunks)} natural paragraph chunks.")
    
    # Print the first three chunks to see the natural boundaries
    for index, chunk in enumerate(chunks[:3], start=1):
        print(f"\nChunk {index}:")
        print(chunk)
        print("-" * 30)