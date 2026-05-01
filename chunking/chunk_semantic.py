from doc_cleaner import clean_doc
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings

def test_semantic_percentiles(target_file):
    print(f"Cleaning {target_file}...")
    raw_text = clean_doc(target_file)
    
    print("Loading embedding model...")
    embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # The whole numbers you want to test
    percentiles_to_test = [70, 75, 80, 85, 90, 95]
    
    print("\n--- Semantic Percentile Grid Search ---")
    
    for p in percentiles_to_test:
        text_splitter = SemanticChunker(
            embedder, 
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=p  # Using whole numbers!
        )
        
        chunks = text_splitter.create_documents([raw_text])
        
        # Calculate average chunk size to see the impact
        avg_size = sum(len(c.page_content) for c in chunks) / len(chunks)
        
        print(f"Percentile {p}: Created {len(chunks)} chunks. (Avg size: ~{int(avg_size)} chars)")

# --- EXECUTION ---
file_to_analyze = "../test/CELEX_32006L0054_IT_TXT.pdf"
test_semantic_percentiles(file_to_analyze)