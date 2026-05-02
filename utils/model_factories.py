import os
from .my_log import logger

from langchain_core.callbacks import BaseCallbackHandler
from ragas.llms import llm_factory

from langchain_community.llms import Ollama
from langchain_core.callbacks import StreamingStdOutCallbackHandler

is_streaming_stdout_enabled = os.getenv("DEBUG_PRINT_STDOUT", "false").lower() == "true"
os.environ["RAGAS_DO_NOT_TRACK"] = 'true'

if is_streaming_stdout_enabled:
    import sys
    import logging
    logging.basicConfig(stream=sys.stdout, level=logging.WARN)
    logging.getLogger("instructor").setLevel(logging.DEBUG)

def create_ollama_model(model, system=None, **kwargs):
    # Initialize the callbacks list from kwargs or a new list
    callbacks = kwargs.pop("callbacks", [])
    
    # Check environment variable
    if is_streaming_stdout_enabled:
        # Only add if it's not already there
        if not any(isinstance(cb, StreamingStdOutCallbackHandler) for cb in callbacks):
            callbacks.append(StreamingStdOutCallbackHandler())
    
    # Return the LangChain Ollama instance
    return Ollama(
        model=model,
        system=system,
        callbacks=callbacks,
        **kwargs
    )

def create_ragas_model(model, provider="openai", **kwargs):

    #if is_streaming_stdout_enabled:
    #    # LiteLLM compatible streaming flag
    #    kwargs["stream"] = True

    return llm_factory(model=model, provider=provider, **kwargs)
