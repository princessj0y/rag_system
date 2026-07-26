import base64
import os
import cairosvg

from typing import List
from langchain_core.documents import Document
from langchain_unstructured import UnstructuredLoader
from langchain_core.messages import HumanMessage

from utils.model_factories import create_ollama_model

def clean_doc(file_path):
    """
    Accepts any file path, auto-detects its type, cleans noise, 
    and returns a list of LangChain Document objects using the modern 1.x loader.
    """

    # to-do: add case for excel/csv 
    if file_path.lower().endswith(('.png', '.jpg', 'jpeg', '.svg')):
        clean_doc = clean_imageful_doc(file_path)
    else:
        clean_doc = clean_textful_doc(file_path)
    
    print(clean_doc)
    return clean_doc

def clean_imageful_doc(file_path):
    """
    Accepts an SVG, PNG, or JPG, converts SVGs in-memory, 
    and uses a local Ollama Vision LLM to extract data into structured Markdown.
    """

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    file_extension = file_path.lower().split('.')[-1]
  
    if file_extension == "svg":
        png_bytes = cairosvg.svg2png(url=file_path, outout_width=1600)
        img_base64 = base64.b64encode(png_bytes).decode("utf-8")
    elif file_extension in ["png", "jpg", "jpeg", "webp"]:
        with open(file_path, "rb") as image_file:
            img_base64 = base64.b64encode(image_file.read()).decode("utf-8")
    else:
        raise ValueError(f"Unsupported file type for vision processing: {file_extension}")

    answer_llm = create_ollama_model(model="gemma4:31b-cloud")

    prompt = """
    You are an expert data extraction assistant. This is an image/infographic filled with statistics.
    Transcribe ALL text, numbers, metrics, and chart data into clean, highly organized Markdown.
    Make sure you explicitly pair metrics with their labels (e.g., 'Passaggi: 1,487 (77% completati)').
    Maintain logical reading order and structure.
    """

    message = HumanMessage(
        content=[
            {"type" : "text", "text": prompt},
            {"type": "image_url", 
             "image_url": {"url": f"data:image/png;base64,{img_base64}"}}
        ]
    )

    response = answer_llm.invoke([message])
    return response.content

def clean_textful_doc(file_path):
    loader = UnstructuredLoader(
        file_path=file_path,
        strategy="hi_res",
        skip_headers_and_footers=True,
        combine_under_n_chars=500
    )
        
    docs = loader.load()
    print(docs)
    if not docs:
        return Document(page_content="", metadata={"source": file_path})

    flattened_text = "\n\n".join([d.page_content for d in docs])
    base_metadata = docs[0].metadata if docs else {}

    with open('cleaned_doc.txt', 'w') as f:
        f.write(flattened_text)

    doc_content = Document(page_content=flattened_text, metadata=base_metadata)
    return doc_content.page_content