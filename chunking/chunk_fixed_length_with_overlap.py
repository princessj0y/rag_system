import nltk
from nltk.tokenize import word_tokenize
from .doc_cleaner import clean_doc

nltk.download('punkt')
nltk.download('punkt_tab') 

def run_overlapping_chunking(raw_text, chunk_size=500, overlap=100, is_eng = False):
    # Overlap cannot be larger than the chunk itself
    if overlap >= chunk_size:
        raise ValueError("Overlap must be strictly less than chunk_size")

    # Set NLTK language
    lang = "english" if is_eng else "italian"

    # Create the chunks
    words = word_tokenize(raw_text, language=lang)
    chunks = [words[i:i + chunk_size] for i in range(0, len(words), chunk_size - overlap)]
    return [" ".join(chunk) for chunk in chunks]

# --- EXECUTION ---
if __name__ == "__main__":
    file_to_analyze = "./test/CELEX_32006L0054_EN_TXT.pdf"
    chunk_size=500
    overlap=100

    raw_text = clean_doc(file_to_analyze)
    chunks = run_overlapping_chunking(raw_text, chunk_size=chunk_size, overlap=overlap)
            
    print(f"\n--- Analysis of {file_to_analyze} ---")
    print(f"Created {len(chunks)} chunks with size {chunk_size} and overlap {overlap}.")
    
    # Print the first two chunks to see the overlap in action
    for index, chunk in enumerate(chunks[:2], start=1):
        print(f"\nChunk {index}:")
        print(chunk)
        print("-" * 30)