from utils.documents import make_chunking_document_aware

def _run_sliding_window(raw_text, window_size=150, overlap=50, is_eng = False):    
    # Split the document into a list of whole words
    # This automatically handles spacing and removes the "half-word" problem
    words = raw_text.split()
    
    chunks = []
    
    # Calculate how far the window slides forward each loop
    step_size = window_size - overlap
    
    # Slide the window across the document
    for i in range(0, len(words), step_size):
        # Grab a "window" of words (e.g., 150 words)
        chunk_words = words[i : i + window_size]
        
        # Glue the words back together into a readable string
        chunk_text = " ".join(chunk_words)
        chunks.append(chunk_text)
        
        # Emergency brake: stop if the window reaches the very end of the word list
        if i + window_size >= len(words):
            break
        
    return chunks

run_sliding_window = make_chunking_document_aware(_run_sliding_window)

# --- EXECUTION ---
if __name__ == "__main__":
    import yaml
    from pathlib import Path
    from .doc_cleaner import clean_doc
    from utils.documents import preserialize_docs

    file_to_analyze = "./test/CELEX_32006L0054_EN_TXT.pdf"
    window_size = 150
    overlap = 50

    raw_text = clean_doc(file_to_analyze)
    chunks = run_sliding_window(raw_text, window_size=window_size, overlap=overlap)

    print(f"\n--- Sliding Window Results ---")
    # print(f"Total Words Found: {len(words)}")
    print(f"Created {len(chunks)} chunks (Window: {window_size} words, Overlap: {overlap} words).")
    
    Path("tmp").mkdir(parents=True, exist_ok=True)
    with open("tmp/chunks-sliding-window.yaml", 'w', encoding='utf-8') as f:
        yaml.dump(preserialize_docs(chunks), f, allow_unicode=True, sort_keys=False)