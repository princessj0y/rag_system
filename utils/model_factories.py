import os
from .my_log import logger
import itertools
import random
import asyncio
import httpx

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

if "GOOGLE_API_KEY" in os.environ:
    model_name = "gemini-3.1-flash-lite-preview"
    _ragas_global_semaphore = asyncio.Semaphore(3)
elif "UNIMI_API_KEY" in os.environ:
    model_name = "Qwen/Qwen3.6-35B-A3B-FP8"
    #model_name = "Qwen/Qwen3-8B"
    _ragas_global_semaphore = asyncio.Semaphore(1)
elif len(ollama_api_keys) > 0:
    model_name = "gpt-oss:120b-cloud"
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
    elif "UNIMI_API_KEY" in os.environ:
        from langchain_openai import ChatOpenAI
        # Merge extra_body if already provided in kwargs
        extra_body = kwargs.pop("extra_body", {})
        extra_body.setdefault("chat_template_kwargs", {"enable_thinking": False})
        llm = ChatOpenAI(
            model=model_name,
            api_key=os.environ.get("UNIMI_API_KEY"),
            base_url="https://open-webui.ricerca.sesar.di.unimi.it/openai",
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
            http_client=httpx.AsyncClient(
                transport=BoundedAsyncHttpxTransport(
                    delegate=httpx.AsyncHTTPTransport(),
                    semaphore=_ragas_global_semaphore
                ),
                timeout=120.0
            )
        )
        return itertools.cycle([create_ragas_model(
            model_name, 
            provider="openai", 
            client=client,
            #stream=True,
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
                    http_client=httpx.AsyncClient(
                        transport=BoundedAsyncHttpxTransport(
                            delegate=AsyncKeyRotationHttpxTransport(shuffled_keys=keys),
                            semaphore=_ragas_global_semaphore
                        ),
                        timeout=120.0,
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
        http_client=httpx.AsyncClient(
            transport=BoundedAsyncHttpxTransport(
                delegate=httpx.AsyncHTTPTransport(),
                semaphore=_ragas_global_semaphore
            ),
            timeout=120.0  # Safe reading window for heavy 120B token generations
        )
    )
    return itertools.cycle([
        create_ragas_model(model_name, provider="openai", client=client) 
    ])

def create_ragas_embedding_model(model, provider="openai", **kwargs):
    from ragas.embeddings.base import embedding_factory

    #if is_streaming_stdout_enabled:
    #    # LiteLLM compatible streaming flag
    #    kwargs["stream"] = True

    return embedding_factory(model=model, provider=provider, interface="modern", **kwargs)

def create_default_embedding_model_iterator():
    models = []
    if "GOOGLE_API_KEY" in os.environ:
        logger.info("Running with Gemini for embeddings...")
        from google import genai
        client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))
        models.append(create_ragas_embedding_model(
            "embeddings_model_name",
            provider="google",
            client=client
        ))
    
    # elif len(ollama_api_keys) > 0:
    #     logger.info(f"Running with Ollama Cloud ({len(ollama_api_keys)} keys found) for embeddings...")
    #     from openai import AsyncOpenAI
    #     for key in ollama_api_keys:
    #         client = AsyncOpenAI(
    #             api_key=key, 
    #             base_url="https://ollama.com/v1"
    #         )
    #         models.append(create_ragas_embedding_model(
    #             embeddings_model_name, 
    #             provider="openai", 
    #             client=client
    #         ))

    else:
        logger.info("Running with Ollama for embeddings...")
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


class SyncKeyRotationHttpxTransport(httpx.HTTPTransport):

    def __init__(self, shuffled_keys, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.keys = shuffled_keys
        self.total_keys = len(shuffled_keys)
        self.current_index = 0

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        while True:
            active_key = self.keys[self.current_index]
            request.headers["Authorization"] = f"Bearer {active_key}"
            try:
                response = super().handle_request(request)
            except Exception as e:
                raise e
            
            if response.status_code != 429:
                return response
            
            self.current_index += 1
            if self.current_index >= self.total_keys:
                raise httpx.HTTPStatusError("Ollama keys completely exhausted.", request=request, response=response)

class AsyncKeyRotationHttpxTransport(httpx.AsyncHTTPTransport):

    def __init__(self, shuffled_keys, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.keys = shuffled_keys
        self.total_keys = len(shuffled_keys)
        self.current_index = 0

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        while True:
            active_key = self.keys[self.current_index]
            request.headers["Authorization"] = f"Bearer {active_key}"
            try:
                response = await super().handle_async_request(request)
            except Exception as e:
                raise e
            
            if response.status_code != 429:
                return response
            
            self.current_index += 1
            if self.current_index >= self.total_keys:
                raise httpx.HTTPStatusError("Ollama keys completely exhausted.", request=request, response=response)

class BoundedAsyncHttpxTransport(httpx.AsyncBaseTransport):
    """
    A decorator wrapper for any httpx Async Transport that chokes concurrency
    using a shared asyncio.Semaphore before delegating the HTTP call.
    """
    def __init__(self, delegate: httpx.AsyncBaseTransport, semaphore: asyncio.Semaphore):
        super().__init__()
        self.delegate = delegate
        self.semaphore = semaphore

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        async with self.semaphore:
            # Delegate the actual HTTP call to the underlying transport instance
            return await self.delegate.handle_async_request(request)

    async def aclose(self) -> None:
        """Ensure the underlying delegate transport is gracefully closed."""
        await self.delegate.aclose()
