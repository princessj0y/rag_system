import pandas # fixes pydantic segfault
import nltk
import json
from pydantic import BaseModel, Field, AliasChoices, AliasPath
from langchain_core.prompts import ChatPromptTemplate

from utils.my_log import logger
from utils.documents import make_chunking_document_aware

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

class ChunkGroupings(BaseModel):
    chunks: list[list[int]] = Field(
        default_factory=list,
        validation_alias=AliasChoices(*[
            alias 
            for k in [ "chunks", "groups", "sentence_groups", "id_chunks", "chunk_list" ]
            # Add nested keys if the model wraps everything in {"properties": {...}}
            for alias in (k, AliasPath("properties", k))
        ]),
        description="A list of lists. Each inner list contains the integer IDs of sentences that form a single logical chunk."
    )

raw_schema_str = json.dumps(ChunkGroupings.model_json_schema(), indent=2)
chunking_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an expert legal document structuring agent.\n"
        "Your task is to group the provided numbered sentences into logical, self-contained chunks.\n"
        "Each chunk should cover exactly one legal concept, recital, or article.\n\n"
        "CRITICAL INSTRUCTION: You MUST return your response as a valid JSON object. "
        "Your JSON response must strictly match this exact JSON schema:\n{schema_json}\n\n"
        "Do not skip any IDs. Every sentence must belong to a chunk."
    ),
    (
        "human",
        "TEXT TO ANALYZE:\n{numbered_text}"
    )
]).partial(schema_json=raw_schema_str)

def _agentic_chunk_block(llm, text_block, is_eng):
    """
    The Agent: Groups numbered sentences into logical chunks.

    Instead of direct text or character bounds, we use numbered sentences:
    1. Zero Source Mutation: If you ask an LLM to return chunked text directly, it will 
       inevitably summarize, paraphrase, or drop transitional words. 
    2. Token vs. Character Blindness: Asking an LLM for character indices (e.g., [0, 150]) 
       is guaranteed to fail. LLMs process tokens, not characters, so they will hallucinate 
       indices that slice words in half.

    To solve this, We programmatically pre-split the text into sentences (via NLTK) and pass 
    them to the LLM with simple integer IDs ([0], [1], etc.). The LLM's only job is logical 
    reasoning: it returns arrays of grouped IDs (e.g., [0, 1, 2]). 
    We then programmatically reconstruct the chunks using the original strings.

    This guarantees 100% text integrity (no lost data). Furthermore, because this function 
    inputs a raw string and outputs exact, unmodified substrings, it slots perfectly into 
    the `make_chunking_document_aware` wrapper. The wrapper's anchor-search can easily map 
    these agentic chunks back to their original Unstructured Document metadata.
    """

    from nltk.tokenize import sent_tokenize

    from pydantic import ValidationError
    from langchain_core.exceptions import OutputParserException
    from langchain_core.messages import AIMessage, HumanMessage
    
    # Pre-split the block into sentences safely
    lang = "english" if is_eng else "italian"
    sentences = sent_tokenize(text_block, language=lang)
    
    if not sentences:
        return []

    # Number the sentences for the LLM
    numbered_text = "\n".join([f"[{i}] {sent}" for i, sent in enumerate(sentences)])
    messages = chunking_prompt.invoke({"numbered_text": numbered_text}).to_messages()
    
    structured_llm = llm.with_structured_output(ChunkGroupings)
    id_chunks = []

    max_attempts = 3
    for attempt in range(max_attempts):
        is_last_attempt = attempt == max_attempts - 1
        try:
            response: ChunkGroupings = structured_llm.invoke(messages)
            id_chunks = response.chunks
            break
            
        except (ValidationError, OutputParserException) as e:
            if is_last_attempt:
                logger.error(f"Agentic chunking schema validation failed after {max_attempts} attempts: {e}")
                break
                
            # Extract raw output and feed the error back to the LLM
            raw_content = getattr(e, "llm_output", None) or getattr(e, "observation", None)
            messages.append(AIMessage(content=str(raw_content) if raw_content else "[Invalid JSON / Schema Output]"))
            messages.append(HumanMessage(
                content=f"Your previous response failed JSON/schema validation with this error:\n{e}\n"
                        f"Please correct your response to strictly match the requested JSON schema."
            ))
            continue
            
        except Exception as e:
            # For network drops, timeouts, or Ollama server crashes, log and bail safely
            logger.exception("General AI Error during metadata generation: %s", e)
            break

    if not id_chunks:
        logger.error("LLM Agent failed. Falling back to original block.")
        return [text_block]

    # Reconstruct the original text using the LLM's ID groupings
    final_text_chunks = []
    # Keep track of used IDs to ensure we don't miss anything if the LLM hallucinates
    used_ids = set()
    
    for id_list in id_chunks:
        # Rebuild the chunk perfectly from the original sentences
        chunk_sentences = []
        for idx in id_list:
            if 0 <= idx < len(sentences) and idx not in used_ids:
                chunk_sentences.append(sentences[idx])
                used_ids.add(idx)
        
        if chunk_sentences:
            final_text_chunks.append(" ".join(chunk_sentences))
            
    # Failsafe: If the LLM forgot some sentences or crashed, make them a new chunk.
    missed_ids = [i for i in range(len(sentences)) if i not in used_ids]
    if missed_ids:
        missed_sentences = [sentences[i] for i in missed_ids]
        final_text_chunks.append(" ".join(missed_sentences))
        
    return final_text_chunks

def _run_agentic_chunking(raw_text, model_name=None, is_eng = False):
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
        # temperature=0.2, num_predict=50
    )
    
    # The Agentic Loop
    final_agentic_chunks = []
    for i, block in enumerate(tqdm(pre_chunks, "Analyzing pre-chunks"), start=1):
        smart_chunks = _agentic_chunk_block(llm, block, is_eng)
        final_agentic_chunks.extend(smart_chunks)

    return final_agentic_chunks

run_agentic_chunking = make_chunking_document_aware(_run_agentic_chunking)

# --- EXECUTION ---
if __name__ == "__main__": 
    import yaml
    from pathlib import Path
    from .doc_cleaner import clean_doc
    from utils.documents import preserialize_docs

    file_to_analyze = "./test/CELEX_32006L0054_EN_TXT.pdf"
    is_eng = 'EN' in file_to_analyze

    raw_text = clean_doc(file_to_analyze)
    # SAFETY: Only taking the first 3000 characters for the test
    # Remove the [:3000] if you want to process the whole document (Warning: Slow!)
    test_text = raw_text[:3000]

    my_chunks = run_agentic_chunking(test_text, is_eng=is_eng, model_name='gpt-oss:120b-cloud')

    print(f"\n--- Agentic Chunking Results ---")
    print(f"The Agent turned raw blocks into {len(my_chunks)} highly refined chunks.")
    
    Path("tmp").mkdir(parents=True, exist_ok=True)    
    with open("tmp/chunks-agentic.yaml", 'w', encoding='utf-8') as f:
        yaml.dump(preserialize_docs(my_chunks), f, allow_unicode=True, sort_keys=False)