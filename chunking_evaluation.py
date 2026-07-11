import re
import csv
import asyncio
import pandas as pd
from datetime import datetime
from tabulate import tabulate
from chunking.doc_cleaner import clean_doc
from dataset.dataset import load_dataset

from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from utils.my_log import logger, mdc

from ragas import experiment, Dataset
from utils.model_factories import model_name, embeddings_model_name, create_default_ragas_model_iterator, create_default_model, default_embeddings
from ragas.metrics.collections import (
    Faithfulness,
    ContextPrecision,
    ContextEntityRecall,
    ContextRecall
)

from chunking.chunk_fixed_length import run_fixed_size_chunking
from chunking.chunk_fixed_length_with_overlap import run_overlapping_chunking
from chunking.chunk_paragraph import run_paragraph_chunking
from chunking.chunk_recursive import run_recursive_chunking
from chunking.chunk_hierarchical import run_hierarchical_chunking
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
    "Hierarchical Legal Chunking": run_recursive_chunking,
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

llm_iterator = create_default_ragas_model_iterator()

def retrieve_chunking_dataset(
    experiment_name,
    chunking_function, 
    raw_text, 
    is_eng, 
    dataset, 
    persist_dir="./chroma_eval_cache"
):
    from langchain_chroma import Chroma
    from langchain_core.documents import Document

    logger.info(f"Targeting persistent collection: {experiment_name}")
    vectorstore = Chroma(
        collection_name=experiment_name,
        embedding_function=default_embeddings,
        persist_directory=persist_dir
    )

    # Check if we already did the work
    if vectorstore._collection.count() == 0:
        logger.info(f"Collection is empty")
        logger.info(f"Performing chunking...")
        raw_chunks = chunking_function(raw_text, is_eng=is_eng)

        processed_docs = []
        for chunk in raw_chunks:
            if isinstance(chunk, str):
                # Wrap old string chunks in a Document
                processed_docs.append(Document(page_content=chunk, metadata={}))
            else:
                # CHROMA FIX: Serialize the list metadata into a string
                if "heading_path" in chunk.metadata and isinstance(chunk.metadata["heading_path"], list):
                    chunk.metadata["heading_path"] = " > ".join(chunk.metadata["heading_path"])
                processed_docs.append(chunk)

        logger.info(f"Performing embedding...")
        vectorstore.add_documents(documents=processed_docs)
    else:
        logger.info(f"Found {vectorstore._collection.count()} chunks! Skipping compute.")

    # Set up the retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    logger.info(f"Evaluating...")
    contexts = []
    for query in dataset["question"]:
        # Per ogni domanda, cerchiamo i 10 pezzi più simili tra i tuoi chunk
        docs = retriever.invoke(query)
        # Salviamo il testo dei pezzi trovati
        contexts.append([d.page_content for d in docs])

    # Aggiungiamo i pezzi trovati al nostro dataset
    dataset["contexts"] = contexts
    dataset["retrieved_contexts"] = contexts
    return dataset

async def evaluate_method(chunking_name, chunking_function, page_index_doc_id, raw_text, is_eng, dataset):
    experiment_name = f"{model_name}_{embeddings_model_name}_{chunking_name.lower().replace(" ", "-")}_{"EN" if is_eng else "IT"}"
    # Replace anything that isn't alphanumeric, dash, or underscore with a dash
    experiment_name = re.sub(r'[^a-zA-Z0-9_-]', '-', experiment_name)
    # Strip leading/trailing punctuation and limit to 63 characters
    experiment_name = experiment_name.strip('_-')[:63]

    if chunking_function == None:
        dataset = retrieve_pageindex_dataset(page_index_doc_id, dataset)
    else:
        dataset = retrieve_chunking_dataset(experiment_name, chunking_function, raw_text, is_eng, dataset)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"{timestamp}_{experiment_name}"

    # Lo usa solo la faithfulness:
    dataset["response"] = []
    answer_llm = create_default_model()
    for i in tqdm(range(len(dataset["question"])), desc="Generating answers"):
        search_prompt = f"""
        Answer only based on provided context.
        Question: {dataset["question"][i]}
        Context: {dataset["contexts"][i]}
        """
        answer = answer_llm.invoke(search_prompt).text
        dataset["response"].append(answer)
    dataset["answer"] = dataset["response"]

    @experiment()
    async def run_rag_evaluation(row):
        cp_score = await ContextPrecision(llm=next(llm_iterator)).ascore(
            user_input=row["user_input"], 
            reference=row["reference"],
            retrieved_contexts=row["retrieved_contexts"],
        )
        cr_score = await ContextRecall(llm=next(llm_iterator)).ascore(
            user_input=row["user_input"], 
            retrieved_contexts=row["retrieved_contexts"],
            reference=row["reference"],
        )
        cer_score = await ContextEntityRecall(llm=next(llm_iterator)).ascore(
            reference=row["reference"], 
            retrieved_contexts=row["retrieved_contexts"],
        )
        f_score = await Faithfulness(llm=next(llm_iterator)).ascore(
            user_input=row["user_input"],
            response=row["response"],
            retrieved_contexts=row["retrieved_contexts"],
        )
        
        return {
            **row,
            "experiment_name": experiment_name,
            "context_precision": cp_score.value,
            "context_recall": cr_score.value,
            "context_entity_recall": cer_score.value,
            "faithfulness": f_score.value
        }

    # Valutazione
    dataset_finale =  Dataset(
        name=experiment_name, 
        backend="local/csv", 
        root_dir=".",
        data=pd.DataFrame(dataset).to_dict(orient="records")
    )
    risultato = await run_rag_evaluation.arun(
        dataset=dataset_finale,
        name=experiment_name,
    )

    logger.info(risultato)
    df = risultato.to_pandas()
    return df['context_precision'].mean(), \
            df['context_recall'].mean(), \
            df['context_entity_recall'].mean(), \
            df['faithfulness'].mean()

async def evaluate_file(file_name, page_index_doc_id):
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
                precision, recall, entity_recall, faithfulness = await evaluate_method(name, chunking_function, page_index_doc_id, raw_text, is_eng, golden_dataset)
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

async def main():
    with logging_redirect_tqdm(loggers=[logger]):
        for (file_name, page_index_doc_id) in tqdm(files, desc="Files"):
            with mdc(file_name=file_name):
                await evaluate_file(file_name, page_index_doc_id)

asyncio.run(main())