import nltk
from nltk.tokenize import sent_tokenize
from doc_cleaner import clean_doc 

# Run this once in your terminal if you haven't: python -m nltk.downloader punkt

def run_advanced_sentence_chunking(target_file, sentences_per_chunk=4, overlap=1):
    print(f"Cleaning {target_file}...")
    
    raw_text = clean_doc(target_file)
    
    # Set NLTK language based on the file name[cite: 1, 2]
    lang = "english" if "EN" in target_file else "italian"
    
    # 2. Use NLTK for smart sentence splitting
    sentences = sent_tokenize(raw_text, language=lang)
    
    chunks = []
    step_size = sentences_per_chunk - overlap
    
    # 3. Use Grouping & Overlap for RAG Context
    for i in range(0, len(sentences), step_size):
        chunk_group = sentences[i : i + sentences_per_chunk]
        chunk_text = " ".join(chunk_group)
        chunks.append(chunk_text)   
        
        if i + sentences_per_chunk >= len(sentences):
            break
            
    print(f"\nCreated {len(chunks)} contextual chunks for the {lang} document.")
        
    return chunks

# --- EXECUTION ---
file_to_analyze = "../test/CELEX_32006L0054_EN_TXT.pdf"
my_chunks = run_advanced_sentence_chunking(file_to_analyze)