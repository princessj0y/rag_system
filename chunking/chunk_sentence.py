import nltk
from utils.documents import make_chunking_document_aware

nltk.download('punkt')
nltk.download('punkt_tab')

def _run_advanced_sentence_chunking(raw_text, sentences_per_chunk=4, overlap=1, is_eng = False):    
    from nltk.tokenize import sent_tokenize
    
    # Set NLTK language
    lang = "english" if is_eng else "italian"
    
    # Use NLTK for smart sentence splitting
    sentences = sent_tokenize(raw_text, language=lang)
    
    chunks = []
    step_size = sentences_per_chunk - overlap
    
    # Use Grouping & Overlap for RAG Context
    for i in range(0, len(sentences), step_size):
        chunk_group = sentences[i : i + sentences_per_chunk]
        chunk_text = " ".join(chunk_group)
        chunks.append(chunk_text)   
        
        if i + sentences_per_chunk >= len(sentences):
            break
        
    return chunks

run_advanced_sentence_chunking = make_chunking_document_aware(_run_advanced_sentence_chunking)

# --- EXECUTION ---
if __name__ == "__main__":
    import yaml
    from pathlib import Path
    from .doc_cleaner import clean_doc
    from utils.documents import preserialize_docs

    file_to_analyze = "./test/CELEX_32006L0054_EN_TXT.pdf"
    
    print(f"Cleaning {file_to_analyze}...")
    raw_text = clean_doc(file_to_analyze)
    
    chunks = run_advanced_sentence_chunking(raw_text)
    print(f"\nCreated {len(chunks)} contextual chunks for the document.")
    
    Path("tmp").mkdir(parents=True, exist_ok=True)
    with open("tmp/chunks-sentence.yaml", 'w', encoding='utf-8') as f:
        yaml.dump(preserialize_docs(chunks), f, allow_unicode=True, sort_keys=False)