import os
import re
import csv
import asyncio
import pandas as pd
from datetime import datetime
from tabulate import tabulate
from chunking.doc_cleaner import clean_doc
from dataset.dataset import load_dataset

from tqdm import tqdm
from utils.my_log import logger, async_mdc

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
    ("./test/CELEX_32006L0054_IT_TXT.pdf", "pi-cmn3q02a805ch0gpk1yqwpuri", 'ita', "./dataset/direttiva_2006_54_REAL_enriched.yaml"),
    ("./test/CELEX_32006L0054_EN_TXT.pdf", "pi-cmn3p5efs00nhlfpka5hmmlto", 'eng', "./dataset/direttiva_2006_54_REAL_enriched_EN.yaml"),
    # ("./test/cross-ref/Kernel.pdf", "pi-cmn3q02a805ch0gpk1yqwpuri", 'eng', "./dataset/cross_referential_dataset.yaml"),
    # ("./test/cross-ref/Operating_system.pdf", "pi-cmn3p5efs00nhlfpka5hmmlto", 'eng', "./dataset/cross_referential_dataset.yaml"),
    # ("./test/cross-ref/Page_fault.pdf", "pi-cmn3p5efs00nhlfpka5hfeato", 'eng', "./dataset/cross_referential_dataset.yaml"),
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
    from utils.documents import flatten_metadata_for_chroma, format_doc_for_llm

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
    logger.info(f"Retrieving contexts...")
    contexts = []
    for query in dataset["question"]:
        # Per ogni domanda, cerchiamo i 10 pezzi più simili tra i tuoi chunk
        docs = retriever.invoke(query)
        # Salviamo il testo dei pezzi trovati
        contexts.append([format_doc_for_llm(d) for d in docs])

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
        joined_context = "\n\n====================\n\n".join(dataset["contexts"][i])
        search_prompt = f"""
        Answer only based on provided context.
        Question: {dataset["question"][i]}
        
        Context: 
        {joined_context}
        """
        answer = answer_llm.invoke(search_prompt).text
        dataset["response"].append(answer)
    dataset["answer"] = dataset["response"]

    @experiment()
    async def run_rag_evaluation(row):
        async def evaluate_cp():
            try:
                score = await ContextPrecision(llm=next(llm_iterator)).ascore(
                    user_input=row["user_input"], 
                    reference=row["reference"],
                    retrieved_contexts=row["retrieved_contexts"]
                )
                return getattr(score, 'value', score)
            except Exception as e:
                logger.warning(f"ContextPrecision failed: {e}")
                return None

        async def evaluate_cr():
            try:
                score = await ContextRecall(llm=next(llm_iterator)).ascore(
                    user_input=row["user_input"], 
                    retrieved_contexts=row["retrieved_contexts"],
                    reference=row["reference"]
                )
                return getattr(score, 'value', score)
            except Exception as e:
                logger.warning(f"ContextRecall failed: {e}")
                return None

        async def evaluate_cer():
            try:
                score = await ContextEntityRecall(llm=next(llm_iterator)).ascore(
                    reference=row["reference"], 
                    retrieved_contexts=row["retrieved_contexts"]
                )
                return getattr(score, 'value', score)
            except Exception as e:
                logger.warning(f"ContextEntityRecall failed: {e}")
                return None

        async def evaluate_f():
            try:
                score = await Faithfulness(llm=next(llm_iterator)).ascore(
                    user_input=row["user_input"],
                    response=row["response"],
                    retrieved_contexts=row["retrieved_contexts"],
                )
                return getattr(score, 'value', score)
            except Exception as e:
                logger.warning(f"Faithfulness failed: {e}")
                return None

        async def evaluate_ns():
            try:
                score = await NoiseSensitivity(llm=next(llm_iterator)).ascore(
                    user_input=row["user_input"],
                    response=row["response"],
                    reference=row["reference"],
                    retrieved_contexts=row["retrieved_contexts"],
                )
                return getattr(score, 'value', score)
            except Exception as e:
                logger.warning(f"NoiseSensitivity failed: {e}")
                return None

        async def evaluate_ar():
            try:
                score = await AnswerRelevancy(llm=next(llm_iterator), embeddings=current_embedding).ascore(
                    user_input=row["user_input"],
                    response=row["response"]
                )
                return getattr(score, 'value', score)
            except Exception as e:
                logger.warning(f"AnswerRelevancy failed: {e}")
                return None

        async def evaluate_ac():
            try:
                score = await AnswerCorrectness(llm=next(llm_iterator), embeddings=current_embedding).ascore(
                    user_input=row["user_input"],
                    response=row["response"],
                    reference=row["reference"]
                )
                return getattr(score, 'value', score)
            except Exception as e:
                logger.warning(f"AnswerCorrectness failed: {e}")
                return None

        # Enqueue ALL metric evaluations for this row at the exact same time.
        # The underlying LLM model impl is in charge of limiting request concurrency via a semaphore, 
        # so that Python can aggressively schedule everything without flooding Ollama/Gemini.
        cp_val, cr_val, cer_val, f_val, ns_val, ar_val, ac_val = await asyncio.gather(
            evaluate_cp(),
            evaluate_cr(),
            evaluate_cer(),
            evaluate_f(),
            evaluate_ns(),
            evaluate_ar(),
            evaluate_ac()
        )

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
    logger.info("Running experiment...")
    risultato = await run_rag_evaluation.arun(
        dataset=dataset_finale,
        name=experiment_name,
    )

    # df = risultato.to_pandas()
    df = df = pd.DataFrame(risultato)
    logger.info(
        f"context_precision: {df['context_precision'].mean()}"
        f", context_recall: {df['context_recall'].mean()}"
        f", context_entity_recall: {df['context_entity_recall'].mean()}"
        f", faithfulness: {df['faithfulness'].mean()}"
        f", noise_sensitivity: {df['noise_sensitivity'].mean()}, "
        f", answer_relevancy: {df['answer_relevancy'].mean()}"
        f", answer_correctness: {df['answer_correctness'].mean()}"
    )
    
    return (
            df['context_precision'].mean(), 
            df['context_recall'].mean(), 
            df['context_entity_recall'].mean(), 
            df['faithfulness'].mean(), 
            df['noise_sensitivity'].mean(), 
            df['answer_relevancy'].mean(), 
            df['answer_correctness'].mean()
    )

async def evaluate_file(file_name, page_index_doc_id, is_eng, dataset_path):
    logger.info(f"Analysing file {file_name} [{page_index_doc_id}]")
    raw_text = clean_doc(file_name, is_eng)

    golden_dataset = load_dataset(dataset_path)

    # Esegui benchmark
    table_data = []
    for name, chunking_function in tqdm(chunking_strategies.items(), desc="Chunking strategies"):
        async with async_mdc(method=name):
            logger.info(f"Metodo {name}...")
            try:
                precision, recall, entity_recall, faithfulness, noise_sensitivity, answer_relevancy, answer_correctness = await evaluate_method(name, chunking_function, page_index_doc_id, raw_text, is_eng, golden_dataset)
                table_data.append([name, f"{precision:.4f}", f"{recall:.4f}", f"{entity_recall:.4f}", f"{faithfulness:.4f}", f"{noise_sensitivity:.4f}", f"{answer_relevancy:.4f}", f"{answer_correctness:.4f}"])
                if faithfulness > 0.75 or faithfulness:
                    logger.info("OK")
                else:
                    logger.info("I'm unable to answer the question")
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
    for (file_name, page_index_doc_id, language, dataset_path) in tqdm(files, desc="Files"):
        async with async_mdc(file_name=file_name):
            await evaluate_file(file_name, page_index_doc_id, (language == 'eng'), dataset_path)

asyncio.run(main())