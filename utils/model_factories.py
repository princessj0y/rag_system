import os
from .my_log import logger
import itertools
import random
import asyncio

import httpx
from .httpx_logging import default_sync_httpx_transport, default_async_httpx_transport
from .httpx_transports import SyncKeyRotationHttpxTransport, AsyncKeyRotationHttpxTransport, BoundedAsyncHttpxTransport

is_streaming_stdout_enabled = os.getenv("DEBUG_PRINT_STDOUT", "false").lower() == "true"
os.environ["RAGAS_DO_NOT_TRACK"] = 'true'

if is_streaming_stdout_enabled:
    import logging
    logging.getLogger("instructor").setLevel(logging.DEBUG)

ollama_api_keys = [
    key for i in range(0, 11)
    if (key := os.environ.get(
        "OLLAMA_API_KEY" if i == 0  else f"OLLAMA_API_KEY_{i}"
    )) is not None
]

########################################################################
#                               LLMs                                   #
########################################################################

timeout = 120.0 # Safe reading window for heavy 120B token generations

if "GOOGLE_API_KEY" in os.environ:
    model_name = "gemini-3.1-flash-lite-preview"
    _ragas_global_semaphore = asyncio.Semaphore(3)
elif "UNIMI_API_KEY" in os.environ:
    model_name = "Qwen/Qwen3.6-35B-A3B-FP8"
    #model_name = "Qwen/Qwen3-8B"
    _ragas_global_semaphore = asyncio.Semaphore(5)
    timeout = 5 * 60
elif len(ollama_api_keys) > 0:
    model_name = "gpt-oss:120b-cloud"
    # model_name = "gemma4"
    _ragas_global_semaphore = asyncio.Semaphore(3 * len(ollama_api_keys))
else:
    model_name = "phi3"
    _ragas_global_semaphore = asyncio.Semaphore(1)

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

class BatchedEmbeddings:
    def __init__(self, embeddings, batch_size=16):
        self.embeddings = embeddings
        self.batch_size = batch_size

    def embed_documents(self, texts):
        if not texts:
            return []
        try:
            return self.embeddings.embed_documents(texts[:self.batch_size]) + self.embed_documents(texts[self.batch_size:])
        except Exception:
            if len(texts) == 1:
                raise
            midpoint = len(texts) // 2
            return self.embed_documents(texts[:midpoint]) + self.embed_documents(texts[midpoint:])

    def embed_query(self, text):
        return self.embeddings.embed_query(text)

if (embeddings_model_name == 'all-MiniLM-L6-v2'
    or embeddings_model_name == 'dlicari/Italian-Legal-BERT'
    or embeddings_model_name == 'nlpaueb/bert-base-uncased-eurlex'):
    from langchain_huggingface import HuggingFaceEmbeddings
    default_embeddings = BatchedEmbeddings(HuggingFaceEmbeddings(model_name=embeddings_model_name))
else:
    from langchain_ollama import OllamaEmbeddings
    default_embeddings = BatchedEmbeddings(OllamaEmbeddings(model=embeddings_model_name))

########################################################################
#                           Factories                                  #
########################################################################

def boundedHttpxAsyncClient(delegate=None):
    return httpx.AsyncClient(
        transport=BoundedAsyncHttpxTransport(
            delegate=delegate, 
            semaphore=_ragas_global_semaphore
        ),
        timeout=timeout
    )

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
    elif "UNIMI_API_KEY" in os.environ:
        from langchain_openai import ChatOpenAI
        if kwargs.pop("format", None) == "json":
            kwargs["response_format"] = {"type": "json_object"}
        kwargs.pop("system", None)
        # Merge extra_body if already provided in kwargs
        extra_body = kwargs.pop("extra_body", {})
        extra_body.setdefault("chat_template_kwargs", {"enable_thinking": False})
        llm = ChatOpenAI(
            model=model_name,
            api_key=os.environ.get("UNIMI_API_KEY"),
            base_url="https://open-webui.ricerca.sesar.di.unimi.it/openai",
            http_client=httpx.Client(transport=default_sync_httpx_transport(), timeout=timeout),
            http_async_client=httpx.AsyncClient(transport=default_async_httpx_transport(), timeout=timeout),
            extra_body=extra_body,
            **kwargs
        )

    elif len(ollama_api_keys) > 0:
        keys = list(ollama_api_keys)
        random.shuffle(keys)
        llm = create_ollama_model(
            model=model_name,
            base_url="https://ollama.com",
            sync_client_kwargs={
                "transport": SyncKeyRotationHttpxTransport(shuffled_keys=keys)
            },
            async_client_kwargs={
                "transport": AsyncKeyRotationHttpxTransport(shuffled_keys=keys)
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

        keys = list(ollama_api_keys)
        random.shuffle(keys)
        return create_ollama_model(
            model=model,
            base_url="https://ollama.com",
            sync_client_kwargs={
                "transport": SyncKeyRotationHttpxTransport(shuffled_keys=keys)
            },
            async_client_kwargs={
                "transport": AsyncKeyRotationHttpxTransport(shuffled_keys=keys)
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
    if "GOOGLE_API_KEY" in os.environ:
        logger.info("Running with Gemini as LLM...")
        # TODO: limit gemini concurrency with the _ragas_global_semaphore
        from google import genai
        client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))
        return itertools.cycle([
            create_ragas_model(model_name, provider="google", client=client)
        ])

    elif "UNIMI_API_KEY" in os.environ:
        logger.info("Running with Unimi as LLM...")
        from openai import AsyncOpenAI
        client = AsyncOpenAI(
            api_key=os.environ.get("UNIMI_API_KEY"), 
            base_url="https://open-webui.ricerca.sesar.di.unimi.it/openai",
            http_client=boundedHttpxAsyncClient()
        )
        return itertools.cycle([create_ragas_model(
            model_name, 
            provider="openai", 
            client=client,
            max_tokens=8192,
            extra_body={
                "chat_template_kwargs": {
                    "enable_thinking": False
                }
            }
        )])
    
    elif len(ollama_api_keys) > 0:
        logger.info(f"Running with Ollama Cloud ({len(ollama_api_keys)} keys found) as LLM...")
        from openai import AsyncOpenAI
        def model_generator():
            while True:
                keys = list(ollama_api_keys)
                random.shuffle(keys)
                client = AsyncOpenAI(
                    api_key="ollama",
                    base_url="https://ollama.com/v1",
                    http_client=boundedHttpxAsyncClient(
                            delegate=AsyncKeyRotationHttpxTransport(shuffled_keys=keys),
                    )
                )
                yield create_ragas_model(
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
                )
        return model_generator()

    logger.info("Running with Ollama as LLM...")
    from openai import AsyncOpenAI
    client = AsyncOpenAI(
        api_key="ollama", 
        base_url="http://localhost:11434/v1",
        http_client=boundedHttpxAsyncClient(),
    )
    return itertools.cycle([
        create_ragas_model(model_name, provider="openai", client=client) 
    ])

def create_ragas_embedding_model(model, provider="openai", **kwargs):
    from ragas.embeddings.base import embedding_factory
    return embedding_factory(model=model, provider=provider, interface="modern", **kwargs)

def create_default_embedding_model_iterator():
    if 'gemini' in embeddings_model_name:
        if "GOOGLE_API_KEY" not in os.environ:
            raise f"no GOOGLE_API_KEY env var found, cannot use {embeddings_model_name}"

        logger.info("Running with Gemini for embeddings...")
        from google import genai
        client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))
        return itertools.cycle([ create_ragas_embedding_model(
            embeddings_model_name,
            provider="google",
            client=client
        ) ])
    
    elif 'cloud' in embeddings_model_name:
        if len(ollama_api_keys) == 0:
            raise f"no OLLAMA_API_KEY env var found, cannot use {embeddings_model_name}"

        logger.info(f"Running with Ollama Cloud ({len(ollama_api_keys)} keys found) for embeddings...")
        from openai import AsyncOpenAI
        def model_generator():
            while True:
                keys = list(ollama_api_keys)
                random.shuffle(keys)
                client = AsyncOpenAI(
                    api_key="ollama", 
                    base_url="https://ollama.com/v1",
                    http_client=boundedHttpxAsyncClient(
                            delegate=AsyncKeyRotationHttpxTransport(shuffled_keys=keys),
                    )
                )
                yield create_ragas_embedding_model(
                    embeddings_model_name, 
                    provider="openai", 
                    client=client,
                    max_tokens=4096,
                    extra_body={
                        "options": {
                            "num_ctx": 8192 # Total context (input + output)
                        }
                    },
                )
            return model_generator()

    logger.info("Running with Ollama for embeddings...")
    return itertools.cycle([
        create_ragas_embedding_model(
            model=f"ollama/{embeddings_model_name}",
            provider="litellm", #todo: why did i choose litellm? 
            api_base="http://localhost:11434",
        )
    ])
