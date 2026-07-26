import pandas # fixes pydantic segfault
from utils.my_log import logger
from utils.documents import make_chunking_document_aware

# --- PROMPTS ---
system_prompt_en = "You are a JSON generator. Output ONLY raw JSON. Use the SAME LANGUAGE as the text to analyze for all values. No intro, no outro, no explanation."
prompt_en = """
    Analyze the following excerpt of an European Directive and return ONLY a JSON file with the following format:
    {{
      "titolo_breve": "a title of max 5 words",
      "riassunto": "a sentence which explains the main legal concept"
    }}
    
    Constraint: All JSON values must be in the same language as the text to analyze below ({text_preview}...).
    
    Text to analyze: {text} 
    """

system_prompt_it = "Sei un generatore di JSON. Produci in output SOLO JSON non formattato. Usa la STESSA LINGUA del testo da analizzare per tutti i valori. Nessuna introduzione, nessuna conclusione, nessuna spiegazione."
prompt_it = """
    Analizza il seguente estratto di una Direttiva Europea e restituisci SOLO un file JSON con il seguente formato:
    {{
      "titolo_breve": "un titolo di massimo 5 parole",
      "riassunto": "una frase che spieghi il concetto legale principale"
    }}
    
    Vincolo: Tutti i valori del JSON devono essere nella stessa lingua del testo da analizzare qui sotto ({text_preview}...).
    
    Testo da analizzare: {text} 
    """

# --- CONFIGURAZIONE OLLAMA ---
def generate_agentic_metadata(llm, text, en):
    """L'Agente analizza il chunk e crea Titolo e Riassunto."""
    import json

    if en:
        prompt = prompt_en.format(text = text[:1000], text_preview = text[:20])
    else:
        prompt = prompt_it.format(text = text[:1000], text_preview = text[:20])
    
    title = "N/A"
    summary = "N/A"

    try:
        response = llm.invoke(prompt).text

        # Parse the non-parsed JSON string
        parsed_metadata = json.loads(response)
        title = parsed_metadata.get("titolo_breve", "N/A")
        summary = parsed_metadata.get("riassunto", "N/A")

    except json.JSONDecodeError as e:
        logger.exception(f"Failed to parse JSON")
    except Exception as e:
        logger.exception(f"Errore AI")

    return title, summary

# --- 2. ESTRAZIONE E CHUNKING ---
def run_agentic_enrich_chunking(docs, model=None, is_eng=False):
    from tqdm import tqdm
    from utils.model_factories import create_model_by_name
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    
    if is_eng:
        system = system_prompt_en
    else:
        system = system_prompt_it

    # aggiungiamo temp e num predict per velocizzare ancora di più
    llm = create_model_by_name(
        model=model,
        format="json",
        system=system,
        # temperature=0.2, num_predict=50
    )

    # Split iniziale (Recursive)
    # Wrap the split_text method so it handles the Unstructured Documents
    document_aware_splitter = make_chunking_document_aware(RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200,
        separators=["\nArticle ", "\n\n", ". "]
    ).split_text)

    chunks = document_aware_splitter(docs)
    
    # Ciclo Agentico: chiediamo a Ollama di "capire" ogni chunk
    for chunk_doc in tqdm(chunks, desc="Enriching chunks"):
        title, summary = generate_agentic_metadata(llm, chunk_doc.page_content, is_eng)
        # Create the enriched chunk
        chunk_doc.page_content = f"TITLE: {title}\nSUMMARY: {summary}\nCONTENT: {chunk_doc.page_content}"
        # Store the generated fields in the metadata
        chunk_doc.metadata["generated_title"] = title
        chunk_doc.metadata["generated_summary"] = summary

    return chunks
    
def run_agentic_enrich_chunking_llama3(pdf_path, is_eng):
    return run_agentic_enrich_chunking('llama3', pdf_path, is_eng)

def run_agentic_enrich_chunking_phi3(pdf_path, is_eng):
    return run_agentic_enrich_chunking('phi3', pdf_path, is_eng) 

def run_agentic_enrich_chunking_gpt_oss(pdf_path, is_eng):
    return run_agentic_enrich_chunking('gpt-oss:120b-cloud', pdf_path, is_eng) 

if __name__ == "__main__":
    import yaml
    from pathlib import Path
    from .doc_cleaner import clean_doc
    from utils.documents import preserialize_docs

    model = 'gpt-oss:120b-cloud'
    FILE_NAME = "./test/CELEX_32006L0054_EN_TXT.pdf"
    is_eng = 'EN' in FILE_NAME

    raw_text = clean_doc(FILE_NAME, is_eng)

    # SAFETY: Only taking the first 3000 characters for the test
    # Remove the [:3000] if you want to process the whole document (Warning: Slow!)
    test_text = raw_text[:3000]

    logger.info(f"Avvio Agentic Chunking su {FILE_NAME}...")
    risultati = run_agentic_enrich_chunking(test_text, model, is_eng)

    Path("tmp").mkdir(parents=True, exist_ok=True)    
    with open("tmp/chunks-agentic-enrich.yaml", 'w', encoding='utf-8') as f:
        yaml.dump(preserialize_docs(risultati), f, allow_unicode=True, sort_keys=False)

    print(f"Operazione completata! Controlla 'tmp/chunks-agentic-enrich.yaml'")