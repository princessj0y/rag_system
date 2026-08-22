import nltk
from utils.documents import make_chunking_document_aware

nltk.download('punkt')
nltk.download('punkt_tab')

def _run_fixed_size_chunking(raw_text, chunk_size=500, is_eng = False):
    from nltk.tokenize import word_tokenize
    
    # Set NLTK language
    lang = "english" if is_eng else "italian"
 
    # Create the chunks
    words = word_tokenize(raw_text, language=lang)
    chunks = [words[i:i + chunk_size] for i in range(0, len(words), chunk_size)]
    return [" ".join(chunk) for chunk in chunks]

run_fixed_size_chunking = make_chunking_document_aware(_run_fixed_size_chunking)

# --- EXECUTION ---
if __name__ == "__main__":
    import yaml
    from pathlib import Path
    from .doc_cleaner import clean_doc
    from utils.documents import preserialize_docs, flatten_metadata_for_chroma

    file_to_analyze = "./test/CELEX_32006L0054_EN_TXT.pdf" 
    raw_text = clean_doc(file_to_analyze)
    chunks = run_fixed_size_chunking(raw_text)

    Path("tmp").mkdir(parents=True, exist_ok=True)    
    with open("tmp/chunks-fixed-len.yaml", 'w', encoding='utf-8') as f:
        yaml.dump(preserialize_docs(chunks), f, allow_unicode=True, sort_keys=False)
    
    for chunk in chunks:
        chunk.metadata = flatten_metadata_for_chroma(chunk.metadata)
    with open("tmp/chunks-fixed-len-flattened.yaml", 'w', encoding='utf-8') as f:
        yaml.dump(preserialize_docs(chunks), f, allow_unicode=True, sort_keys=False)