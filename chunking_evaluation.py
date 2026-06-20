import csv
from dataset.dataset import load_dataset
from tabulate import tabulate
from datasets import Dataset
from langchain_community.vectorstores import Chroma
from chunking.doc_cleaner import clean_doc

from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from utils.my_log import logger, mdc

from langchain_ollama import OllamaEmbeddings

from ragas import evaluate
from utils.model_factories import create_default_ragas_model_iterator, create_default_model
from ragas.metrics._context_precision import ContextPrecision
from ragas.metrics._context_recall import ContextRecall
from ragas.metrics._context_entities_recall import ContextEntityRecall
from ragas.metrics._faithfulness import Faithfulness

from chunking.chunk_fixed_length import run_fixed_size_chunking
from chunking.chunk_fixed_length_with_overlap import run_overlapping_chunking
from chunking.chunk_paragraph import run_paragraph_chunking
from chunking.chunk_recursive import run_recursive_chunking
from chunking.chunk_semantic import run_semantic_chunking_70, run_semantic_chunking_75, run_semantic_chunking_80, run_semantic_chunking_85, run_semantic_chunking_90
from chunking.chunk_sentence import run_advanced_sentence_chunking
from chunking.chunk_sliding_window import run_sliding_window
from chunking.chunk_agentic import run_agentic_chunking
from chunking.chunk_agentic_enrich import run_agentic_enrich_chunking

from page_index.pageindex_retriever import retrieve_dataset as retrieve_pageindex_dataset

files = [
    #("./test/CELEX_32006L0054_IT_TXT.pdf", "pi-cmn3q02a805ch0gpk1yqwpuri"),
    ("./test/CELEX_32006L0054_EN_TXT.pdf", "pi-cmn3p5efs00nhlfpka5hmmlto"),
]

# Metodi di chunking
# Keys are labels, Values are the actual function objects
chunking_strategies = {
    "Fixed Length Chunking": run_fixed_size_chunking,
    "Fixed Length Chunking With Overlap": run_overlapping_chunking,
    "Paragraph-based Chunking": run_paragraph_chunking,
    "Recursive Chunking": run_recursive_chunking,
    "Semantic Chunking 0.70": run_semantic_chunking_70,
    "Semantic Chunking 0.75": run_semantic_chunking_75,
    "Semantic Chunking 0.80": run_semantic_chunking_80,
    "Semantic Chunking 0.85": run_semantic_chunking_85,
    "Semantic Chunking 0.90": run_semantic_chunking_90,
    "Sentence-based Chunking": run_advanced_sentence_chunking,
    "Sliding Window Chunking": run_sliding_window,
    #"Agentic Chunking Phi3": process_agentic_phi3,
    #"Agentic Chunking llama3": process_agentic_llama3,
    "Agentic Chunking gpt-oss": run_agentic_chunking,
    "Agentic Enrich Chunking gpt-oss": run_agentic_enrich_chunking,
    "PageIndex": None,
}

# Assicurati di aver fatto 'ollama pull nomic-embed-text' nel terminale
embeddings = OllamaEmbeddings(model='qwen3-embedding:0.6b')
#embeddings = OllamaEmbeddings(model="mxbai-embed-large")
llm_iterator = create_default_ragas_model_iterator()

def retrieve_chunking_dataset(chunking_function, raw_text, is_eng, dataset):
    # Esegui il chunking
    logger.info(f"Performing chunking...")
    chunks = chunking_function(raw_text, is_eng=is_eng)

    # Creiamo il database temporaneo con i TUOI chunk
    logger.info(f"Creating temporary vector store using Chroma...")
    contexts = []
    
    # Chroma accepts chunks and embeddings.
    # We add an ephemeral collection name so it clears/overwrites properly in memory.
    vectorstore = Chroma.from_texts(
        texts=chunks, 
        embedding=embeddings,
        collection_name="temp_eval_collection"
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    logger.info(f"Evaluating...")
    for query in dataset["question"]:
        # Per ogni domanda, cerchiamo i 3 pezzi più simili tra i tuoi chunk
        docs = retriever.invoke(query)
        # Salviamo il testo dei pezzi trovati
        contexts.append([d.page_content for d in docs])

    # Pulizia: Cancella la collezione per evitare che i chunk si mischino al prossimo ciclo
    vectorstore.delete_collection()

    # Aggiungiamo i pezzi trovati al nostro dataset
    dataset["contexts"] = contexts
    dataset["retrieved_contexts"] = contexts
    return dataset

def evaluate_method(name, chunking_function, page_index_doc_id, raw_text, is_eng, dataset):
    
    if chunking_function == None:
        dataset = retrieve_pageindex_dataset(page_index_doc_id, dataset)
    else:
        dataset = retrieve_chunking_dataset(chunking_function, raw_text, is_eng, dataset)

    # Lo usa solo la faithfulness:
    dataset["response"] = []
    answer_llm = create_default_model()
    for i in range(len(dataset["question"])):
        search_prompt = f"""
        Answer only based on provided context.
        Question: {dataset["question"][i]}
        Context: {dataset["contexts"][i]}
        """
        answer = answer_llm.invoke(search_prompt).text
        dataset["response"].append(answer)
    dataset["answer"] = dataset["response"]

    # Valutazione
    dataset_finale = Dataset.from_dict(dataset)
    risultato = evaluate(
        dataset_finale, 
        embeddings=embeddings,
        metrics=[
            ContextPrecision(llm=next(llm_iterator)), 
            ContextRecall(llm=next(llm_iterator)), 
            ContextEntityRecall(llm=next(llm_iterator)), 
            Faithfulness(llm=next(llm_iterator)),
        ], 
    )

    logger.info(risultato)
    df = risultato.to_pandas()
    return df['context_precision'].mean(), \
            df['context_recall'].mean(), \
            df['context_entity_recall'].mean(), \
            df['faithfulness'].mean()

with logging_redirect_tqdm(loggers=[logger]):
    for (file_name, page_index_doc_id) in tqdm(files, desc="Files"):
        with mdc(file_name=file_name):
            logger.info(f"Analysing file {file_name} [{page_index_doc_id}]")
            raw_text = clean_doc(file_name)
            is_eng = 'EN' in file_name

            if is_eng:
                golden_dataset = load_dataset("./dataset/direttiva_2006_54_REAL_enriched_EN.yaml")
            else:
                golden_dataset = load_dataset("./dataset/direttiva_2006_54_REAL_enriched.yaml")

            # Esegui benchmark
            table_data = []
            for name, chunking_function in tqdm(chunking_strategies.items(), desc="Chunking strategies"):
                with mdc(method=name):
                    logger.info(f"Metodo {name}...")
                    try:
                        precision, recall, entity_recall, faithfulness = evaluate_method(name, chunking_function, page_index_doc_id, raw_text, is_eng, golden_dataset)
                        table_data.append([name, f"{precision:.4f}", f"{recall:.4f}", f"{entity_recall:.4f}", f"{faithfulness:.4f}"])
                    except Exception as e:
                        logger.exception("Failed, skipping")

            headers = ["Method", "Precision", "Recall","ContextEntitiesRecall", "Faithfullness"]
            tqdm.write("\n" + tabulate(table_data, headers=headers, tablefmt="fancy_grid"))

            with open(file_name + '.csv', 'w', newline='') as csvfile:
                csvwriter = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL)
                csvwriter.writerow(headers)
                for row in table_data:
                    csvwriter.writerow(row)
