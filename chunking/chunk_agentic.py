import pandas # fixes pydantic segfault
from utils.my_log import logger

def agentic_chunk_block(llm, text_block):
    """The Agent: Asks Ollama to logically split a block of text."""
    
    prompt = f"""
    You are an expert legal document structuring agent.
    Your task is to read the following text and split it into logical, self-contained chunks.
    Each chunk should cover exactly one legal concept, recital, or article.
    
    TEXT TO ANALYZE:
    {text_block}
    
    OUTPUT FORMAT:
    You must respond ONLY with a valid JSON object. Do not add markdown formatting, do not say "Here is the JSON".
    Strictly use this format:
    {{
        "chunks": [
            "first logical chunk here",
            "second logical chunk here"
        ]
    }}
    """
    import json
    
    try:
        response = llm.invoke(prompt).text
        # Parse the JSON string back into a Python dictionary
        data = json.loads(response)
        return data.get("chunks", [])
    except json.JSONDecodeError:
        logger.exception("LLM Agent failed. Falling back to original block.")
        return [text_block]

def run_agentic_chunking(raw_text, model_name=None, is_eng = False):
    from tqdm import tqdm
    from utils.model_factories import create_model_by_name
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    logger.info("Pre-chunking text to feed the Agent...")
    # The Pre-Chunker (creates digestible blocks for the LLM)
    pre_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=0)
    pre_chunks = pre_splitter.split_text(raw_text)
    
    logger.info(f"Sending {len(pre_chunks)} blocks to the Ollama Agent. Please wait...")

    system = "You output strictly valid JSON."
    llm = create_model_by_name(
        model=model_name,
        format="json",
        system=system,
        # temperature=0.2, num_predict=50
    )
    
    # The Agentic Loop
    final_agentic_chunks = []
    for i, block in enumerate(tqdm(pre_chunks, "Analyzing pre-chunks"), start=1):
        smart_chunks = agentic_chunk_block(llm, block)
        final_agentic_chunks.extend(smart_chunks)

    return final_agentic_chunks

# --- EXECUTION ---
if __name__ == "__main__": 
    from .doc_cleaner import clean_doc

    file_to_analyze = "./test/CELEX_32006L0054_EN_TXT.pdf"
    raw_text = clean_doc(file_to_analyze)

    # SAFETY: Only taking the first 3000 characters for the test
    # Remove the [:3000] if you want to process the whole document (Warning: Slow!)
    test_text = raw_text[:3000]

    my_chunks = run_agentic_chunking(test_text, model='gpt-oss:120b-cloud')

    print(f"\n--- Agentic Chunking Results ---")
    print(f"The Agent turned raw blocks into {len(my_chunks)} highly refined chunks.")
    
    for index, chunk in enumerate(my_chunks[:4], start=1):
        print(f"\n AGENT CHUNK {index}:")
        print(chunk)
        print("-" * 30)