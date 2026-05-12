import os
from .my_log import logger

from langchain_core.callbacks import BaseCallbackHandler
from ragas.llms import llm_factory

from langchain_ollama import ChatOllama
from langchain_core.callbacks import StreamingStdOutCallbackHandler

from google import genai
from openai import OpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

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
    return ChatOllama(
        model=model,
        system=system,
        callbacks=callbacks,
        **kwargs
    )

def create_default_model(**kwargs):
    if "GOOGLE_API_KEY" in os.environ:
        llm = ChatGoogleGenerativeAI(
            model="gemini-3.1-flash-lite-preview",
            google_api_key=os.environ.get("GOOGLE_API_KEY"),
            **kwargs
        )
    elif "OLLAMA_API_KEY" in os.environ:
        llm = create_ollama_model(
            model="gpt-oss:120b-cloud",
            base_url="https://ollama.com",
            headers={"Authorization": f"Bearer {os.environ.get("OLLAMA_API_KEY")}"},
            **kwargs
        )
    else:
        llm = create_ollama_model(
            model="phi3",
            **kwargs
        )
    return llm 

        
def create_ragas_model(model, provider="openai", **kwargs):

    #if is_streaming_stdout_enabled:
    #    # LiteLLM compatible streaming flag
    #    kwargs["stream"] = True

    return llm_factory(model=model, provider=provider, **kwargs)

def create_default_ragas_model():
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
    
    return llm 