import os
import csv
import numpy as np
from dataset.dataset import load_dataset
from utils.my_log import logger, mdc
from tabulate import tabulate
from datasets import Dataset
from langchain_community.vectorstores import FAISS
from chunking.doc_cleaner import clean_doc

from langchain_ollama import OllamaEmbeddings

from google import genai
from openai import OpenAI
from ragas import evaluate
from utils.model_factories import create_ragas_model
from ragas.metrics._context_precision import ContextPrecision
from ragas.metrics._context_recall import ContextRecall

from chunking.chunk_fixed_length import run_fixed_size_chunking
from chunking.chunk_fixed_length_with_overlap import run_overlapping_chunking
from chunking.chunk_paragraph import run_paragraph_chunking
from chunking.chunk_recursive import run_recursive_chunking
from chunking.chunk_semantic import run_semantic_chunking_70, run_semantic_chunking_75, run_semantic_chunking_80, run_semantic_chunking_85, run_semantic_chunking_90
from chunking.chunk_sentence import run_advanced_sentence_chunking
from chunking.chunk_sliding_window import run_sliding_window
from chunking.chunk_agentic import run_agentic_chunking
from chunking.chunk_agentic_enrich import run_agentic_enrich_chunking

files = [
    "./test/CELEX_32006L0054_IT_TXT.pdf",
    #"./test/CELEX_32006L0054_EN_TXT.pdf",
]

# Metodi di chunking
# Keys are labels, Values are the actual function objects
chunking_strategies = {
    #"Fixed Length Chunking": run_fixed_size_chunking,
    #"Fixed Length Chunking With Overlap": run_overlapping_chunking,
    #"Paragraph-based Chunking": run_paragraph_chunking,
    #"Recursive Chunking": run_recursive_chunking,
    #"Semantic Chunking 0.70": run_semantic_chunking_70,
    #"Semantic Chunking 0.75": run_semantic_chunking_75,
    #"Semantic Chunking 0.80": run_semantic_chunking_80,
    #"Semantic Chunking 0.85": run_semantic_chunking_85,
    #"Semantic Chunking 0.90": run_semantic_chunking_90,
    #"Sentence-based Chunking": run_advanced_sentence_chunking,
    #"Sliding Window Chunking": run_sliding_window,
    #"Agentic Chunking Phi3": process_agentic_phi3,
    #"Agentic Chunking llama3": process_agentic_llama3,
    #"Agentic Chunking gpt-oss": run_agentic_chunking,
    "Agentic Enrich Chunking gpt-oss": run_agentic_enrich_chunking,
}

# Assicurati di aver fatto 'ollama pull nomic-embed-text' nel terminale
embeddings = OllamaEmbeddings(model="nomic-embed-text")
#embeddings = OllamaEmbeddings(model="mxbai-embed-large")

if "GOOGLE_API_KEY" in os.environ:
    logger.info("Running with Gemini...")
    client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))
    llm = create_ragas_model(
        "gemini-3.1-flash-lite-preview",
        provider="google",
        client=client
    )
elif "OLLAMA_API_KEY" in os.environ:
    logger.info("Running with Ollama Cloud...")
    client = OpenAI(
        api_key=os.environ.get("OLLAMA_API_KEY"), 
        base_url="https://ollama.com/v1"
    )
    llm = create_ragas_model(
        "gpt-oss:120b-cloud", 
        provider="openai", 
        client=client,                  
        max_tokens=4096, 
        # Ollama-specific context window size
        extra_body={
            "options": {
                "num_ctx": 8192 # Total context (input + output)
            }
        },
    )
else:
    logger.info("Running with Ollama...")
    client = OpenAI(
        api_key="ollama", 
        base_url="http://localhost:11434/v1"
    )
    llm = create_ragas_model("phi3", provider="openai", client=client)

def evaluate_method(name, chunking_function, raw_text, is_eng, dataset):
    # Esegui il chunking
    logger.info(f"Performing chunking...")
    chunks = chunking_function(raw_text, is_eng=is_eng)

    # Creiamo il database temporaneo con i TUOI chunk
    logger.info(f"Creating temporary vector store...")
    contexts = []
    vectorstore = FAISS.from_texts(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    logger.info(f"Evaluating...")
    for query in dataset["question"]:
        # Per ogni domanda, cerchiamo i 3 pezzi più simili tra i tuoi chunk
        docs = retriever.invoke(query)
        # Salviamo il testo dei pezzi trovati
        contexts.append([d.page_content for d in docs])

    # Aggiungiamo i pezzi trovati al nostro dataset
    dataset["contexts"] = contexts

    # Valutazione
    dataset_finale = Dataset.from_dict(dataset)
    risultato = evaluate(
        dataset_finale, 
        embeddings=embeddings,
        metrics=[ContextPrecision(llm=llm), ContextRecall(llm=llm)],
    )
    logger.info(risultato)
    return float(np.mean(risultato['context_precision'])), \
            float(np.mean(risultato['context_recall']))

for file_name in files:
    with mdc(file_name=file_name):
        logger.info(f"Analysing file {file_name}")
        raw_text = clean_doc(file_name)
        is_eng = 'EN' in file_name

        if is_eng:
            golden_dataset = load_dataset("./dataset/direttiva_2006_54_REAL_enriched.yaml")
        else:
            golden_dataset = load_dataset("./dataset/direttiva_2006_54_REAL_enriched.yaml")

        # Esegui benchmark
        table_data = []
        for name, chunking_function in chunking_strategies.items():
            with mdc(method=name):
                logger.info(f"Metodo {name}...")
                try:
                    precision, recall = evaluate_method(name, chunking_function, raw_text, is_eng, golden_dataset)
                    table_data.append([name, f"{precision:.4f}", f"{recall:.4f}"])
                except Exception as e:
                    logger.exception("Failed, skipping")

        headers = ["Method", "Precision", "Recall"]
        print("\n" + tabulate(table_data, headers=headers, tablefmt="fancy_grid"))

        with open(file_name + '.csv', 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL)
            csvwriter.writerow(headers)
            for row in table_data:
                csvwriter.writerow(row)