import os
from .my_log import logger
import itertools

from langchain_core.callbacks import BaseCallbackHandler
from ragas.llms import llm_factory

from langchain_ollama import ChatOllama
from langchain_core.callbacks import StreamingStdOutCallbackHandler

from google import genai
from openai import AsyncOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

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

if "GOOGLE_API_KEY" in os.environ:
    model_name = "gemini-3.1-flash-lite-preview"
elif ollama_api_keys_cycle is not None:
    model_name = "gpt-oss:120b-cloud"
else:
    model_name = "phi3"

def create_ollama_model(model, system=None, **kwargs):
    # Initialize the callbacks list from kwargs or a new list
    callbacks = kwargs.pop("callbacks", [])
    
    # Check environment variable
    if is_streaming_stdout_enabled:
        # Only add if it's not already there
        if not any(isinstance(cb, StreamingStdOutCallbackHandler) for cb in callbacks):
            callbacks.append(StreamingStdOutCallbackHandler())
    
    # Return the LangChain Ollama instance
    return ChatOllama(
        model=model,
        system=system,
        callbacks=callbacks,
        **kwargs
    )

def create_default_model(**kwargs):
    if "GOOGLE_API_KEY" in os.environ:
        llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=os.environ.get("GOOGLE_API_KEY"),
            **kwargs
        )
    elif ollama_api_keys_cycle is not None:
        llm = create_ollama_model(
            model=model_name,
            base_url="https://ollama.com",
            headers={"Authorization": f"Bearer {next(ollama_api_keys_cycle)}"},
            **kwargs
        )
    else:
        llm = create_ollama_model(
            model=model_name,
            **kwargs
        )
    return llm 

        
def create_ragas_model(model, provider="openai", **kwargs):

    #if is_streaming_stdout_enabled:
    #    # LiteLLM compatible streaming flag
    #    kwargs["stream"] = True

    return llm_factory(model=model, provider=provider, **kwargs)

def create_default_ragas_model_iterator():
    models = []
    if "GOOGLE_API_KEY" in os.environ:
        logger.info("Running with Gemini...")
        client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))
        models.append(create_ragas_model(
            model_name,
            provider="google",
            client=client
        ))
    
    elif len(ollama_api_keys) > 0:
        logger.info(f"Running with Ollama Cloud ({len(ollama_api_keys)} keys found)...")
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
        client = AsyncOpenAI(
            api_key="ollama", 
            base_url="http://localhost:11434/v1"
        )
        models.append(create_ragas_model(model_name, provider="openai", client=client))

    return itertools.cycle(models)
