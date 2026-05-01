import os
from doc_cleaner import clean_doc
from langchain_text_splitters import RecursiveCharacterTextSplitter

def run_recursive_chunking(target_file, chunk_size=1200, overlap=150):
    print(f"\n{'='*50}")
    print(f"Processing: {os.path.basename(target_file)}")
    print(f"{'='*50}")
    
    # 1. Extract the clean Markdown text
    raw_text = clean_doc(target_file)
    
    # 2. Initialize the Recursive Splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len,
        # The Waterfall: Paragraphs -> Line breaks -> Sentences -> Words -> Characters
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    # 3. Create the chunks
    chunks = text_splitter.split_text(raw_text)
    
    print(f"✅ Created {len(chunks)} chunks.")
    print(f"Target Max Size: {chunk_size} chars | Safety Overlap: {overlap} chars\n")
    
    # 4. Print the first 3 chunks to evaluate the boundaries
    for index, chunk in enumerate(chunks[:3], start=1):
        print(f"--- CHUNK {index} ({len(chunk)} characters) ---")
        print(chunk)
        print("-" * 50)
        
    return chunks

# --- EXECUTION ---
# Make sure these paths match your folder structure!
file_en = "../test/CELEX_32006L0054_EN_TXT.pdf"
file_it = "../test/CELEX_32006L0054_IT_TXT.pdf"

# Run the English file first
chunks_en = run_recursive_chunking(file_en, chunk_size=1200, overlap=150)

# Run the Italian file second to compare
chunks_it = run_recursive_chunking(file_it, chunk_size=1200, overlap=150)