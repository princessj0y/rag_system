
def run_semantic_chunking(raw_text, threshold, is_eng = False):
    from utils.model_factories import default_embeddings
    from langchain_experimental.text_splitter import SemanticChunker

    text_splitter = SemanticChunker(
        default_embeddings, 
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=threshold
    )
        
    documents = text_splitter.create_documents([raw_text])
    return documents

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
    from .doc_cleaner import clean_doc
    
    # The whole numbers you want to test
    percentiles_to_test = [70, 75, 80, 85, 90, 95]
    #file_to_analyze = "./test/CELEX_32006L0054_EN_TXT.pdf"
    file_to_analyze = "./test/CELEX_32006L0054_IT_TXT.pdf"
    raw_text = clean_doc(file_to_analyze)
    
    print("\n--- Semantic Percentile Grid Search ---")
    for p in percentiles_to_test:
        documents = run_semantic_chunking(raw_text, p)
        chunks = [d.page_content for d in documents]

        # Calculate average chunk size to see the impact
        avg_size = sum(len(c) for c in chunks) / len(chunks)
            
        print(f"Percentile {p}: Created {len(chunks)} chunks. (Avg size: ~{int(avg_size)} chars)")