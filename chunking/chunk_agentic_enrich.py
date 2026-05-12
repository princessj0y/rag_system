import os
import json
from .doc_cleaner import clean_doc
from utils.my_log import logger
from utils.model_factories import create_ollama_model
from langchain_text_splitters import RecursiveCharacterTextSplitter

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
        title = parsed_metadata.get("title", "N/A")
        summary = parsed_metadata.get("summary", "N/A")

    except json.JSONDecodeError as e:
        logger.exception(f"Failed to parse JSON")
    except Exception as e:
        logger.exception(f"Errore AI")

    return title, summary

# --- 2. ESTRAZIONE E CHUNKING ---
def run_agentic_enrich_chunking(raw_text, model="gpt-oss:120b-cloud", is_eng=False):
    
    if is_eng:
        system = system_prompt_en
    else:
        system = system_prompt_it

    # aggiungiamo temp e num predict per velocizzare ancora di più
    if 'cloud' not in model:
        llm = create_ollama_model(
            model=model,
            format="json",
            system=system,
            # temperature=0.2, num_predict=50
        )
    else:
        if "OLLAMA_API_KEY" not in os.environ:
            raise "no OLLAMA_API_KEY env var found"
        
        llm = create_ollama_model(
            model=model,
            format="json",
            system=system,
            base_url="https://ollama.com",
            headers={"Authorization": f"Bearer {os.environ.get("OLLAMA_API_KEY")}"},
        )

    # Split iniziale (Recursive)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200,
        separators=["\nArticle ", "\n\n", ". "]
    )
    initial_chunks = splitter.split_text(raw_text)
    
    agentic_results = []
    
    # Ciclo Agentico: chiediamo a Ollama di "capire" ogni chunk
    for i, chunk in enumerate(initial_chunks):
        logger.info(f"Analisi chunk {i+1}/{len(initial_chunks)}...")
        title, summary = generate_agentic_metadata(llm, chunk, is_eng)
        # Create the enriched chunk
        agentic_results.append(f"TITLE: {title}\nSUMMARY: {summary}\nCONTENT: {chunk}")
        logger.info("")
        
    return agentic_results
    
def run_agentic_enrich_chunking_llama3(pdf_path, is_eng):
    return run_agentic_enrich_chunking('llama3', pdf_path, is_eng)

def run_agentic_enrich_chunking_phi3(pdf_path, is_eng):
    return run_agentic_enrich_chunking('phi3', pdf_path, is_eng) 

def run_agentic_enrich_chunking_gpt_oss(pdf_path, is_eng):
    return run_agentic_enrich_chunking('gpt-oss:120b-cloud', pdf_path, is_eng) 

if __name__ == "__main__":
    model = 'gpt-oss:120b-cloud'
    FILE_NAME = "CELEX_32006L0054_EN_TXT.pdf"
    raw_text = clean_doc(FILE_NAME)

    # SAFETY: Only taking the first 3000 characters for the test
    # Remove the [:3000] if you want to process the whole document (Warning: Slow!)
    test_text = raw_text[:3000]

    logger.info(f"Avvio Agentic Chunking su {FILE_NAME}...")
    risultati = run_agentic_enrich_chunking(test_text, model, 'EN' in FILE_NAME)

    with open(f"analisi_agentic_{model}.txt", "w", encoding="utf-8") as f:
        for item in risultati:
            f.write(f"=== CHUNK {item['id']} ===\n")
            f.write(f"ANALISI AGENTE:\n{item['ai_analysis']}\n")
            f.write(f"TESTO ORIGINALE:\n{item['content']}\n\n")

    print(f"Operazione completata! Controlla 'analisi_agentic_{model}.txt'")