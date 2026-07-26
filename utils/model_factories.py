import os
from .my_log import logger
import itertools
import random

is_streaming_stdout_enabled = os.getenv("DEBUG_PRINT_STDOUT", "false").lower() == "true"
os.environ["RAGAS_DO_NOT_TRACK"] = 'true'

if is_streaming_stdout_enabled:
    import sys
    import logging
    logging.basicConfig(stream=sys.stdout, level=logging.WARN)
    logging.getLogger("instructor").setLevel(logging.DEBUG)

ollama_api_keys = [
    key for i in range(0, 11)
    if (key := os.environ.get(
        "OLLAMA_API_KEY" if i == 0  else f"OLLAMA_API_KEY_{i}"
    )) is not None
]
ollama_api_keys_cycle = itertools.cycle(ollama_api_keys) if ollama_api_keys else None

# Randomize the first model used by skipping a random amount
for i in range(random.randint(0, len(ollama_api_keys))):
    next(ollama_api_keys_cycle)

########################################################################
#                               LLMs                                   #
########################################################################

if "GOOGLE_API_KEY" in os.environ:
    model_name = "gemini-3.1-flash-lite-preview"
elif ollama_api_keys_cycle is not None:
    model_name = "gpt-oss:120b-cloud"
else:
    model_name = "phi3"

########################################################################
#                           EMBEDDINGS                                 #
########################################################################

# Per i modelli Ollama, assicurati di aver fatto 'ollama pull <modello>' nel terminale
#embeddings_model_name = 'snowflake-arctic-embed2'
embeddings_model_name = 'qwen3-embedding:0.6b'
#embeddings_model_name = 'nomic-embed-text'
#embeddings_model_name = 'mxbai-embed-large'

#embeddings_model_name = 'dlicari/Italian-Legal-BERT'
#embeddings_model_name = 'nlpaueb/bert-base-uncased-eurlex'
#embeddings_model_name = 'all-MiniLM-L6-v2'

if (embeddings_model_name == 'all-MiniLM-L6-v2'
    or embeddings_model_name == 'dlicari/Italian-Legal-BERT'
    or embeddings_model_name == 'nlpaueb/bert-base-uncased-eurlex'):
    from langchain_huggingface import HuggingFaceEmbeddings
    default_embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
else:
    from langchain_ollama import OllamaEmbeddings
    default_embeddings = OllamaEmbeddings(model=embeddings_model_name)

def create_ollama_model(model, system=None, **kwargs):
    # Initialize the callbacks list from kwargs or a new list
    callbacks = kwargs.pop("callbacks", [])
    
    # Check environment variable
    if is_streaming_stdout_enabled:
        # Only add if it's not already there
        from langchain_core.callbacks import StreamingStdOutCallbackHandler
        if not any(isinstance(cb, StreamingStdOutCallbackHandler) for cb in callbacks):
            callbacks.append(StreamingStdOutCallbackHandler())
    
    # Return the LangChain Ollama instance
    from langchain_ollama import ChatOllama
    return ChatOllama(
        model=model,
        system=system,
        callbacks=callbacks,
        **kwargs
    )

def create_default_model(**kwargs):
    if "GOOGLE_API_KEY" in os.environ:
        from langchain_google_genai import ChatGoogleGenerativeAI
        llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=os.environ.get("GOOGLE_API_KEY"),
            **kwargs
        )
    elif ollama_api_keys_cycle is not None:
        llm = create_ollama_model(
            model=model_name,
            base_url="https://ollama.com",
            client_kwargs={
                "headers": {
                    "Authorization": f"Bearer {next(ollama_api_keys_cycle)}"
                }
            },
            **kwargs
        )
    else:
        llm = create_ollama_model(
            model=model_name,
            **kwargs
        )
    return llm 

def create_model_by_name(model, **kwargs):
    if model is None:
        return create_default_model(**kwargs)
    
    if 'gemini' in model:
        if "GOOGLE_API_KEY" not in os.environ:
            raise f"no GOOGLE_API_KEY env var found, cannot use {model}"
        
        from langchain_google_genai import ChatGoogleGenerativeAI
        llm = ChatGoogleGenerativeAI(
            model=model,
            google_api_key=os.environ.get("GOOGLE_API_KEY"),
            **kwargs
        )

    elif 'cloud' in model:
        if len(ollama_api_keys) == 0:
            raise f"no OLLAMA_API_KEY env var found, cannot use {model}"

        return create_ollama_model(
            model=model,
            base_url="https://ollama.com",
            client_kwargs={
                "headers": {
                    "Authorization": f"Bearer {next(ollama_api_keys_cycle)}"
                }
            },
            **kwargs
        )
    
    return create_ollama_model(
        model=model,
        base_url="http://127.0.0.1:11434",
        **kwargs
    )
        
def create_ragas_model(model, provider="openai", **kwargs):
    from ragas.llms import llm_factory

    #if is_streaming_stdout_enabled:
    #    # LiteLLM compatible streaming flag
    #    kwargs["stream"] = True

    return llm_factory(model=model, provider=provider, **kwargs)

def create_default_ragas_model_iterator():
    models = []
    if "GOOGLE_API_KEY" in os.environ:
        logger.info("Running with Gemini...")
        from google import genai
        client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))
        models.append(create_ragas_model(
            model_name,
            provider="google",
            client=client
        ))
    
    elif len(ollama_api_keys) > 0:
        logger.info(f"Running with Ollama Cloud ({len(ollama_api_keys)} keys found)...")
        from openai import AsyncOpenAI
        for key in ollama_api_keys:
            client = AsyncOpenAI(
                api_key=key, 
                base_url="https://ollama.com/v1"
            )
            models.append(create_ragas_model(
                model_name, 
                provider="openai", 
                client=client,                  
                max_tokens=4096, 
                # Ollama-specific context window size
                extra_body={
                    "options": {
                        "num_ctx": 8192 # Total context (input + output)
                    }
                },
            ))

    else:
        logger.info("Running with Ollama...")
        from openai import AsyncOpenAI
        client = AsyncOpenAI(
            api_key="ollama", 
            base_url="http://localhost:11434/v1"
        )
        models.append(create_ragas_model(model_name, provider="openai", client=client))

    iter = itertools.cycle(models)
    # Randomize the first model used by skipping a random amount
    for i in range(random.randint(0, len(models))):
        next(iter)
    return iter

def create_ragas_embedding_model(model, provider="openai", **kwargs):
    from ragas.embeddings.base import embedding_factory

    #if is_streaming_stdout_enabled:
    #    # LiteLLM compatible streaming flag
    #    kwargs["stream"] = True

    return embedding_factory(model=model, provider=provider, interface="modern", **kwargs)

def create_default_embedding_model_iterator():
    models = []
    if "GOOGLE_API_KEY" in os.environ:
        logger.info("Running with Gemini...")
        from google import genai
        client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))
        models.append(create_ragas_embedding_model(
            "embeddings_model_name",
            provider="google",
            client=client
        ))
    
    # elif len(ollama_api_keys) > 0:
    #     logger.info(f"Running with Ollama Cloud ({len(ollama_api_keys)} keys found)...")
    #     from openai import AsyncOpenAI
    #     for key in ollama_api_keys:
    #         client = AsyncOpenAI(
    #             api_key=key, 
    #             base_url="https://ollama.com"
    #         )
    #         models.append(create_ragas_embedding_model(
    #             embeddings_model_name, 
    #             provider="openai", 
    #             client=client
    #         ))

    else:
        logger.info("Running with Ollama...")
        # from langchain_ollama import OllamaEmbeddings
        # from ragas.embeddings import LangchainEmbeddingsWrapper

        # client = OllamaEmbeddings(
        #     model=embeddings_model_name,
        #     base_url="http://localhost:11434"
        # )
        # ragas_embedding = LangchainEmbeddingsWrapper(client)
        # models.append(ragas_embedding)
        from ragas.embeddings.base import embedding_factory
        ragas_embedding = embedding_factory(
            provider="litellm",
            model=f"ollama/{embeddings_model_name}",  # e.g., "ollama/nomic-embed-text"
            api_base="http://localhost:11434",
            interface="modern"
        )
        models.append(ragas_embedding)

    iter = itertools.cycle(models)
    # Randomize the first model used by skipping a random amount
    for i in range(random.randint(0, len(models))):
        next(iter)
    return iter


