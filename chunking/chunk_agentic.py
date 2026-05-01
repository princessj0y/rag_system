import json
import ollama
from doc_cleaner import clean_doc
from langchain_text_splitters import RecursiveCharacterTextSplitter

def agentic_chunk_block(text_block):
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
    
    # We use llama3, but you can change this to mistral or whatever you have pulled
    response = ollama.chat(model='llama3', messages=[
        {'role': 'system', 'content': 'You output strictly valid JSON.'},
        {'role': 'user', 'content': prompt}
    ])
    
    response_text = response['message']['content']
    
    try:
        # Parse the JSON string back into a Python dictionary
        data = json.loads(response_text)
        return data.get("chunks", [])
    except json.JSONDecodeError:
        print("⚠️ LLM Agent failed to return valid JSON. Falling back to original block.")
        return [text_block]

def run_agentic_chunking(target_file):
    print(f"Cleaning {target_file}...")
    raw_text = clean_doc(target_file)
    
    # SAFETY: Only taking the first 3000 characters for the test
    # Remove the [:3000] if you want to process the whole document (Warning: Slow!)
    test_text = raw_text[:3000]
    
    print("Pre-chunking text to feed the Agent...")
    # 1. The Pre-Chunker (creates digestible blocks for the LLM)
    pre_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=0)
    pre_chunks = pre_splitter.split_text(test_text)
    
    final_agentic_chunks = []
    
    print(f"Sending {len(pre_chunks)} blocks to the Ollama Agent. Please wait...")
    
    # 2. The Agentic Loop
    for i, block in enumerate(pre_chunks, start=1):
        print(f"Agent is analyzing block {i}...")
        smart_chunks = agentic_chunk_block(block)
        final_agentic_chunks.extend(smart_chunks)
        
    print(f"\n--- Agentic Chunking Results ---")
    print(f"The Agent turned {len(pre_chunks)} raw blocks into {len(final_agentic_chunks)} highly refined chunks.")
    
    for index, chunk in enumerate(final_agentic_chunks[:4], start=1):
        print(f"\n AGENT CHUNK {index}:")
        print(chunk)
        print("-" * 30)

    return final_agentic_chunks

# --- EXECUTION ---
file_to_analyze = "../test/CELEX_32006L0054_EN_TXT.pdf"
my_chunks = run_agentic_chunking(file_to_analyze)