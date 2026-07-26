from utils.documents import make_chunking_document_aware

def _run_paragraph_chunking(raw_text, is_eng = False):
    
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

run_paragraph_chunking = make_chunking_document_aware(_run_paragraph_chunking)

# --- EXECUTION ---
if __name__ == "__main__":
    import yaml
    from pathlib import Path
    from .doc_cleaner import clean_doc
    from utils.documents import preserialize_docs
    
    file_to_analyze = "./test/CELEX_32006L0054_IT_TXT.pdf"
    raw_text = clean_doc(file_to_analyze)
    chunks = run_paragraph_chunking(raw_text)
            
    print(f"\n--- Paragraph Chunking Results ---")
    print(f"Created {len(chunks)} natural paragraph chunks.")
    
    Path("tmp").mkdir(parents=True, exist_ok=True)    
    with open("tmp/chunks-paragraph.yaml", 'w', encoding='utf-8') as f:
        yaml.dump(preserialize_docs(chunks), f, allow_unicode=True, sort_keys=False)