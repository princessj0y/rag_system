from .doc_cleaner import clean_doc
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings

def run_semantic_chunking(raw_text, threshold, is_eng = False):
    embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    text_splitter = SemanticChunker(
        embedder, 
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=threshold  # Using whole numbers!
    )
        
    documents = text_splitter.create_documents([raw_text])
    return [d.page_content for d in documents]

def run_semantic_chunking_70(raw_text, is_eng = False):
    return run_semantic_chunking(raw_text, 70, is_eng)

def run_semantic_chunking_75(raw_text, is_eng = False):
    return run_semantic_chunking(raw_text, 75, is_eng)

def run_semantic_chunking_80(raw_text, is_eng = False):
    return run_semantic_chunking(raw_text, 80, is_eng)

def run_semantic_chunking_85(raw_text, is_eng = False):
    return run_semantic_chunking(raw_text, 85, is_eng)

def run_semantic_chunking_90(raw_text, is_eng = False):
    return run_semantic_chunking(raw_text, 90, is_eng)

# --- EXECUTION ---
if __name__ == "__main__":
    # The whole numbers you want to test
    percentiles_to_test = [70, 75, 80, 85, 90, 95]
    file_to_analyze = "./test/CELEX_32006L0054_IT_TXT.pdf"
    raw_text = clean_doc(file_to_analyze)
    
    print("\n--- Semantic Percentile Grid Search ---")
    for p in percentiles_to_test:
        chunks = run_semantic_chunking(raw_text, p)

        # Calculate average chunk size to see the impact
        avg_size = sum(len(c) for c in chunks) / len(chunks)
            
        print(f"Percentile {p}: Created {len(chunks)} chunks. (Avg size: ~{int(avg_size)} chars)")