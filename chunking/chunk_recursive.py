import pandas # fixes pydantic segfault
from utils.documents import make_chunking_document_aware

def _run_recursive_chunking(raw_text, chunk_size=1200, overlap=150, is_eng = False):
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    if is_eng:
        separators=["\nTITLE ", "\nCHAPTER ", "\nSection ", "\nArticle ", "\n\n", "\n", " "]
    else:
        separators=["\nTITOLO ", "\nCAPITOLO ", "\nSezione ", "\nArticolo ", "\n\n", "\n", " "]

    # Initialize the Recursive Splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len,     
        separators=separators
    )
    
    # Create the chunks
    chunks = text_splitter.split_text(raw_text)
        
    return chunks

run_recursive_chunking = make_chunking_document_aware(_run_recursive_chunking)

if __name__ == "__main__":
    import os
    import yaml
    from pathlib import Path
    from .doc_cleaner import clean_doc
    from utils.documents import preserialize_docs

    file_en = "./test/CELEX_32006L0054_EN_TXT.pdf"
    file_it = "./test/CELEX_32006L0054_IT_TXT.pdf"
    Path("tmp").mkdir(parents=True, exist_ok=True)

    chunk_size = 1200
    overlap = 150

    # Run the English file first
    print(f"\n{'='*50}")
    print(f"Processing: {os.path.basename(file_en)}")
    print(f"{'='*50}")

    raw_text = clean_doc(file_en)
    chunks_en = run_recursive_chunking(raw_text, chunk_size=chunk_size, overlap=overlap)
    
    print(f"Created {len(chunks_en)} chunks.")
    print(f"Target Max Size: {chunk_size} chars | Safety Overlap: {overlap} chars\n")
     
    with open("tmp/chunks-recursive-en.yaml", 'w', encoding='utf-8') as f:
        yaml.dump(preserialize_docs(chunks_en), f, allow_unicode=True, sort_keys=False)

    # Run the Italian file second to compare
    print(f"\n{'='*50}")
    print(f"Processing: {os.path.basename(file_it)}")
    print(f"{'='*50}")

    raw_text = clean_doc(file_it)
    chunks_it = run_recursive_chunking(raw_text, chunk_size=chunk_size, overlap=overlap)
    
    print(f"Created {len(chunks_it)} chunks.")
    print(f"Target Max Size: {chunk_size} chars | Safety Overlap: {overlap} chars\n")
    
    with open("tmp/chunks-recursive-it.yaml", 'w', encoding='utf-8') as f:
        yaml.dump(preserialize_docs(chunks_it), f, allow_unicode=True, sort_keys=False)