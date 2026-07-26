import os
import re
import csv
import asyncio
import pandas as pd
from datetime import datetime
from tabulate import tabulate
from chunking.doc_cleaner import clean_doc
from dataset.dataset import load_dataset
from utils.documents import flatten_metadata_for_chroma

from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from utils.my_log import logger, mdc

from ragas import experiment, Dataset
from utils.model_factories import model_name, embeddings_model_name, create_default_ragas_model_iterator, create_default_model, default_embeddings, create_default_embedding_model_iterator
from ragas.metrics.collections import (
    Faithfulness,
    ContextPrecision,
    ContextEntityRecall,
    ContextRecall,
    NoiseSensitivity,
    AnswerRelevancy,
    AnswerCorrectness
)

from chunking.chunk_fixed_length import run_fixed_size_chunking
from chunking.chunk_fixed_length_with_overlap import run_overlapping_chunking
from chunking.chunk_paragraph import run_paragraph_chunking
from chunking.chunk_recursive import run_recursive_chunking
from chunking.chunk_hierarchical_legal import run_hierarchical_legal_chunking
from chunking.chunk_semantic import run_semantic_chunking_70, run_semantic_chunking_75, run_semantic_chunking_80, run_semantic_chunking_85, run_semantic_chunking_90
from chunking.chunk_sentence import run_advanced_sentence_chunking
from chunking.chunk_sliding_window import run_sliding_window
from chunking.chunk_agentic import run_agentic_chunking
from chunking.chunk_agentic_enrich import run_agentic_enrich_chunking

from page_index.pageindex_retriever import retrieve_dataset as retrieve_pageindex_dataset

files = [
    ("./test/CELEX_32006L0054_IT_TXT.pdf", "pi-cmn3q02a805ch0gpk1yqwpuri"),
    ("./test/CELEX_32006L0054_EN_TXT.pdf", "pi-cmn3p5efs00nhlfpka5hmmlto"),
]

# Metodi di chunking
# Keys are labels, Values are the actual function objects
chunking_strategies = {
    "Fixed Length Chunking": run_fixed_size_chunking,
    "Fixed Length Chunking With Overlap": run_overlapping_chunking,
    "Paragraph-based Chunking": run_paragraph_chunking,
    "Recursive Chunking": run_recursive_chunking,
    "Hierarchical Legal Chunking": run_hierarchical_legal_chunking,
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
embedding_iterator = create_default_embedding_model_iterator()
current_embedding = next(embedding_iterator)

def retrieve_chunking_dataset(
    experiment_name,
    chunking_function, 
    raw_text, 
    is_eng, 
    dataset, 
    base_persist_dir="./chroma_eval_cache"
):
    from langchain_chroma import Chroma
    from langchain_core.documents import Document

    experiment_dir = os.path.join(base_persist_dir, experiment_name)
    logger.info(f"Targeting persistent directory: {experiment_dir}")
    vectorstore = Chroma(
        collection_name="eval_collection",
        embedding_function=default_embeddings,
        persist_directory=experiment_dir
    )

    # Check if we already did the work
    if vectorstore._collection.count() == 0:
        logger.info(f"Collection is empty")
        logger.info(f"Performing chunking...")
        raw_chunks = chunking_function(raw_text, is_eng=is_eng)

        processed_docs = []
        for chunk in raw_chunks:
            chunk.metadata = flatten_metadata_for_chroma(chunk.metadata)
            processed_docs.append(chunk)

        logger.info(f"Performing embedding...")
        vectorstore.add_documents(documents=processed_docs)
    else:
        logger.info(f"Found {vectorstore._collection.count()} chunks! Skipping compute.")

    # Set up the retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
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
    # Strip leading/trailing punctuation
    experiment_name = experiment_name.strip('_-')

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
        cp_val = cr_val = cer_val = f_val = ns_val = ar_val = ac_val = None

        try:
            cp_score = await ContextPrecision(llm=next(llm_iterator)).ascore(
                user_input=row["user_input"], 
                reference=row["reference"],
                retrieved_contexts=row["retrieved_contexts"]
            )
            cp_val = getattr(cp_score, 'value', cp_score)
        except Exception as e:
            logger.warning(f"ContextPrecision failed: {e}")

        try:
            cr_score = await ContextRecall(llm=next(llm_iterator)).ascore(
                user_input=row["user_input"], 
                retrieved_contexts=row["retrieved_contexts"],
                reference=row["reference"]
            )
            cr_val = getattr(cr_score, 'value', cr_score)
        except Exception as e:
            logger.warning(f"ContextRecall failed: {e}")

        try:
            cer_score = await ContextEntityRecall(llm=next(llm_iterator)).ascore(
                reference=row["reference"], 
                retrieved_contexts=row["retrieved_contexts"]
            )
            cer_val = getattr(cer_score, 'value', cer_score)
        except Exception as e:
            logger.warning(f"ContextEntityRecall failed: {e}")

        try:
            f_score = await Faithfulness(llm=next(llm_iterator)).ascore(
                user_input=row["user_input"],
                response=row["response"],
                retrieved_contexts=row["retrieved_contexts"],
            )
            f_val = getattr(f_score, 'value', f_score)
        except Exception as e:
            logger.warning(f"Faithfulness failed: {e}")

        try:
            ns_score = await NoiseSensitivity(llm=next(llm_iterator)).ascore(
                user_input=row["user_input"],
                response=row["response"],
                reference=row["reference"],
                retrieved_contexts=row["retrieved_contexts"],
            )
            ns_val = getattr(ns_score, 'value', ns_score)
        except Exception as e:
            logger.warning(f"NoiseSensitivity failed: {e}")

        try:
            ar_score = await AnswerRelevancy(llm=next(llm_iterator), embeddings=current_embedding).ascore(
                user_input=row["user_input"],
                response=row["response"]
            )
            ar_val = getattr(ar_score, 'value', ar_score)
        except Exception as e:
            logger.warning(f"AnswerRelevancy failed: {e}")

        try:
            ac_score = await AnswerCorrectness(llm=next(llm_iterator), embeddings=current_embedding).ascore(
                user_input=row["user_input"],
                response=row["response"],
                reference=row["reference"]
            )
            ac_val = getattr(ac_score, 'value', ac_score)
        except Exception as e:
            logger.warning(f"AnswerCorrectness failed: {e}")

        return {
            **row,
            "experiment_name": experiment_name,
            "context_precision": cp_val,
            "context_recall": cr_val,
            "context_entity_recall": cer_val,
            "faithfulness": f_val,
            "noise_sensitivity": ns_val,
            "answer_relevancy": ar_val,
            "answer_correctness": ac_val
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
    risultato = [await run_rag_evaluation(row) for row in dataset_finale]

    logger.info(risultato)
    # df = risultato.to_pandas()
    df = df = pd.DataFrame(risultato)
    return (
            df['context_precision'].mean(), 
            df['context_recall'].mean(), 
            df['context_entity_recall'].mean(), 
            df['faithfulness'].mean(), 
            df['noise_sensitivity'].mean(), 
            df['answer_relevancy'].mean(), 
            df['answer_correctness'].mean()
    )

async def evaluate_file(file_name, page_index_doc_id):
    logger.info(f"Analysing file {file_name} [{page_index_doc_id}]")
    raw_text = clean_doc(file_name)
    is_eng = True #'EN' in file_name

    golden_dataset = load_dataset("./dataset/cross_referential_dataset.yaml")

    # Esegui benchmark
    table_data = []
    for name, chunking_function in tqdm(chunking_strategies.items(), desc="Chunking strategies"):
        with mdc(method=name):
            logger.info(f"Metodo {name}...")
            try:
                precision, recall, entity_recall, faithfulness, noise_sensitivity, answer_relevancy, answer_correctness = await evaluate_method(name, chunking_function, page_index_doc_id, raw_text, is_eng, golden_dataset)
                table_data.append([name, f"{precision:.4f}", f"{recall:.4f}", f"{entity_recall:.4f}", f"{faithfulness:.4f}", f"{noise_sensitivity:.4f}", f"{answer_relevancy:.4f}", f"{answer_correctness:.4f}"])
                if faithfulness > 0.75 or faithfulness:
                    print("OK")
                else:
                    print("I'm unable to answer the question")
            except Exception as e:
                logger.exception("Failed, skipping")

    headers = ["Method", "Precision", "Recall","ContextEntitiesRecall", "Faithfullness", "NoiseSensitivty", "AnswerRelevancy", "AnswerCorrectness"]
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