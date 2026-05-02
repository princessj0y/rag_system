import nltk
from nltk.tokenize import sent_tokenize
from .doc_cleaner import clean_doc 

nltk.download('punkt')
nltk.download('punkt_tab')

def run_advanced_sentence_chunking(target_file, sentences_per_chunk=4, overlap=1, is_eng = False):    
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

# --- EXECUTION ---
if __name__ == "__main__":
    file_to_analyze = "./test/CELEX_32006L0054_EN_TXT.pdf"
    
    print(f"Cleaning {file_to_analyze}...")
    raw_text = clean_doc(file_to_analyze)
    
    chunks = run_advanced_sentence_chunking(raw_text)
    print(f"\nCreated {len(chunks)} contextual chunks for the document.")