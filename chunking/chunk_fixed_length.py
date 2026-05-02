import nltk
from nltk.tokenize import word_tokenize
from .doc_cleaner import clean_doc

nltk.download('punkt')
nltk.download('punkt_tab')

def run_fixed_size_chunking(raw_text, chunk_size=500, is_eng = False):   
    # Set NLTK language
    lang = "english" if is_eng else "italian"
 
    # Create the chunks
    words = word_tokenize(raw_text, language=lang)
    chunks = [words[i:i + chunk_size] for i in range(0, len(words), chunk_size)]
    return [" ".join(chunk) for chunk in chunks]

# --- EXECUTION ---
if __name__ == "__main__":
    file_to_analyze = "./test/CELEX_32006L0054_EN_TXT.pdf" 
    raw_text = clean_doc(file_to_analyze)
    chunks = run_fixed_size_chunking(raw_text)
    
    # Print the chunks with labels
    print(f"\n--- Analysis of {file_to_analyze} ---")
    for index, chunk in enumerate(chunks, start=1):
        print(f"Chunk {index}:")
        print(chunk)
        print("-" * 30) # Separator line for readability