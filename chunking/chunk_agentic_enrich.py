import pandas # fixes pydantic segfault
from utils.my_log import logger
from utils.documents import make_chunking_document_aware
from pydantic import BaseModel, Field, AliasChoices, AliasPath
from langchain_core.prompts import ChatPromptTemplate
import json

class ChunkMetadata(BaseModel):
    titolo_breve: str = Field(
        validation_alias=AliasChoices(*[
            alias 
            for k in [
                # Italian variants
                "titolo_breve", "titolo", "titolo_sintetico",
                # English variants
                "short_title", "title", "brief_title",
            ]
            # Add nested keys if the model wraps everything in {"properties": {...}}
            for alias in (k, AliasPath("properties", k))
        ]),
        description="A short title of max 5 words / Un titolo di massimo 5 parole.",
    )
    riassunto: str = Field(
        validation_alias=AliasChoices(*[
            alias 
            for k in [
                # Italian variants
                "riassunto", "sommario", "concetto_legale", "descrizione",
                # English variants
                "summary", "abstract", "legal_concept", "brief_summary", "description"
            ]
            # Add nested keys if the model wraps everything in {"properties": {...}}
            for alias in (k, AliasPath("properties", k))
        ]),
        description="A single sentence explaining the main legal concept / Una frase che spieghi il concetto legale principale.",
    )

raw_schema_str = json.dumps(ChunkMetadata.model_json_schema(), indent=2)
prompt_en = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an analytical assistant. You analyze legal text and extract metadata.\n"
        "CRITICAL INSTRUCTION: You MUST return your response as a valid JSON object. "
        "Your JSON response must strictly match this exact JSON schema:\n{schema_json}\n\n"
        "Constraint: Use the SAME LANGUAGE as the text below for all JSON values. No intro, no outro, no explanation."
    ),
    (
        "human",
        "Analyze this excerpt from a European Directive:\n\n{text}"
    )
]).partial(schema_json=raw_schema_str)

prompt_it = ChatPromptTemplate.from_messages([
    (
        "system",
        "Sei un assistente analitico. Analizzi testi legali ed estrai metadati.\n"
        "ISTRUZIONE CRITICA: DEVI restituire la risposta come oggetto JSON valido. "
        "La tua risposta JSON deve corrispondere rigorosamente a questo schema JSON:\n{schema_json}\n\n"
        "Vincolo: Usa la STESSA LINGUA del testo qui sotto per tutti i valori JSON. Nessuna introduzione, nessuna conclusione, nessuna spiegazione."
    ),
    (
        "human",
        "Analizza questo estratto di una Direttiva Europea:\n\n{text}"
    )
]).partial(schema_json=raw_schema_str)

# --- CONFIGURAZIONE OLLAMA ---
def generate_agentic_metadata(llm, text, en):
    """L'Agente analizza il chunk e crea Titolo e Riassunto."""
    from pydantic import ValidationError
    from langchain_core.exceptions import OutputParserException
    from langchain_core.messages import AIMessage, HumanMessage
    
    structured_llm = llm.with_structured_output(ChunkMetadata)
    prompt_template = prompt_en if en else prompt_it
    messages = prompt_template.invoke({"text": text[:1000]}).to_messages()

    max_attempts = 3
    for attempt in range(max_attempts):
        is_last_attempt = attempt == max_attempts - 1

        try:
            response: ChunkMetadata = structured_llm.invoke(messages)
            return response.titolo_breve, response.riassunto
            
        except (ValidationError, OutputParserException) as e:
            if is_last_attempt:
                logger.exception(f"Schema validation failed after {max_attempts} attempts: %s", e)
                break # Exit loop and return N/A
                
            # Extract raw output and feed the error back to the LLM
            raw_content = getattr(e, "llm_output", None) or getattr(e, "observation", None)
            messages.append(AIMessage(content=str(raw_content) if raw_content else "[Invalid JSON / Schema Output]"))
            
            error_lang = "English" if en else "Italian"
            messages.append(HumanMessage(
                content=f"Your previous response failed JSON/schema validation with this error:\n{e}\n"
                        f"Please correct your response to strictly match the requested JSON schema in {error_lang}."
            ))
            continue
            
        except Exception as e:
            # For network drops, timeouts, or Ollama server crashes, log and bail safely
            logger.exception("General AI Error during metadata generation: %s", e)
            break

    return "N/A", "N/A"

# --- 2. ESTRAZIONE E CHUNKING ---
def run_agentic_enrich_chunking(docs, model=None, is_eng=False):
    from tqdm import tqdm
    from utils.model_factories import create_model_by_name
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    # aggiungiamo temp e num predict per velocizzare ancora di più
    llm = create_model_by_name(
        model=model,
        format="json",
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