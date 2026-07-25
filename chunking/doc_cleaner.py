import os
import yaml
from pathlib import Path

from utils.model_factories import create_model_by_name

from tqdm import tqdm
from utils.my_log import logger

def clean_doc(file_path, is_eng=False):
    """
    Accepts any file path, auto-detects its type, cleans noise, 
    and returns a list of LangChain Document objects using the modern 1.x loader.
    """

    # to-do: add case for excel/csv 
    if file_path.lower().endswith(('.png', '.jpg', 'jpeg', '.svg')):
        clean_doc = clean_imageful_doc(file_path)
    else:
        clean_doc = clean_textful_doc(file_path, is_eng)
    
    print(clean_doc)
    return clean_doc

# Tell PyYAML to use block scalars (|) for multiline strings for ultimate readability
def str_presenter(dumper, data):
    if len(data.splitlines()) > 1:
        return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='|')
    return dumper.represent_scalar('tag:yaml.org,2002:str', data)
    
yaml.add_representer(str, str_presenter)

def clean_textful_doc(file_path, is_eng):
    from langchain_unstructured import UnstructuredLoader
    from langchain_core.documents import Document
    
    file_hash = get_file_hash(file_path)

    # Setup cache directory
    cache_dir = Path(f"./cleaned_docs_cache/{file_hash}")
    cache_dir.mkdir(parents=True, exist_ok=True)

    image_dir = cache_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    
    cache_file = cache_dir / "docs.yaml"

    # Check cached hit
    if cache_file.exists():
        logger.info(f"Cache hit! Loading parsed document from {cache_file}")
        with open(cache_file, 'r', encoding='utf-8') as f:
            cached_data = yaml.full_load(f)
        # Reconstruct LangChain Documents
        docs = [Document(**doc_dict) for doc_dict in cached_data]
    else:
        logger.info(f"Cache miss. Running Unstructured 'hi_res' on {file_path}...")
        loader = UnstructuredLoader(
            file_path=file_path,
            strategy="hi_res",
            # https://tesseract-ocr.github.io/tessdoc/Data-Files-in-different-versions.html
            languages=['eng' if is_eng else 'ita'],
            skip_headers_and_footers=True, # strips page numbers, repetitive document titles at the top of pages, and legal footers
            # rip images and save them
            extract_image_block_types=["Image"],
            extract_image_block_output_dir=str(image_dir)
        )
        
        docs = loader.load()
        if not docs:
            docs = [ Document(page_content="", metadata={"source": file_path}) ]

        # Langchain documents use Pydantic under the hood. 
        # model_dump() works for v2, dict() for v1.
        # Sanitize the dictionary to remove NumPy types and Tuples
        docs_dict = sanitize_for_yaml([
            doc.model_dump() if hasattr(doc, 'model_dump') else doc.dict() 
            for doc in docs
        ])
        
        with open(cache_file, 'w', encoding='utf-8') as f:
            yaml.dump(docs_dict, f, allow_unicode=True, sort_keys=False)
        logger.info(f"Saved parsed output to {cache_file}")

    # Enrich images
    image_docs = [doc for doc in docs if doc.metadata.get("category") == "Image"]
    if image_docs:
        for doc in tqdm(image_docs, desc="Processing extracted images"):
            img_path = doc.metadata.get("image_path")
            if img_path and os.path.exists(img_path):
                vision_text = clean_imageful_doc(img_path, cache_path=Path(img_path).with_suffix('.txt'))
                # 1. Put it in page_content so it gets embedded and searched
                doc.page_content = f"Image Description:\n{vision_text}"    
                # 2. Tuck a safe copy in the metadata just in case it gets chopped
                doc.metadata["raw_payload"] = vision_text
                doc.metadata["payload_type"] = "image_markdown"
    
    # Handle tables
    for doc in docs:
        category = doc.metadata.get("category")
        if not category == "Table":
            continue
        
        html_table = doc.metadata.get("text_as_html")
        if not html_table:
            continue
        
        # Leave doc.page_content exactly as Unstructured made it (flattened text)
        # because it's great for vector search keywords.
                
        # Stuff the fragile HTML into the metadata
        doc.metadata["raw_payload"] = html_table
        doc.metadata["payload_type"] = "table_html"

    return docs

def clean_imageful_doc(file_path, cache_path=None):
    """
    Accepts an SVG, PNG, or JPG, converts SVGs in-memory, 
    and uses a local Ollama Vision LLM to extract data into structured Markdown.
    """
    import base64
    from langchain_core.messages import HumanMessage

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # Fallback to a hash-based cache file next to the original image if none is provided
    if cache_path is None:
        file_hash = get_file_hash(file_path)
        cache_path = Path(file_path).with_name(f"{file_hash}.txt")
    cache_path = Path(cache_path)

    # Check cache
    if cache_path.exists():
        logger.info(f"Vision cache hit! Loading from {cache_path}")
        with open(cache_path, 'r', encoding='utf-8') as f:
            return f.read()

    logger.info(f"Cache miss. Running vision model on {file_path}...")
    file_extension = file_path.lower().split('.')[-1]
  
    if file_extension == "svg":
        import cairosvg
        png_bytes = cairosvg.svg2png(url=file_path, outout_width=1600)
        img_base64 = base64.b64encode(png_bytes).decode("utf-8")
    elif file_extension in ["png", "jpg", "jpeg", "webp"]:
        with open(file_path, "rb") as image_file:
            img_base64 = base64.b64encode(image_file.read()).decode("utf-8")
    else:
        raise ValueError(f"Unsupported file type for vision processing: {file_extension}")

    answer_llm = create_model_by_name(model="gemma4:31b-cloud")

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
    vision_text = response.content

    # Save to cache
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, 'w', encoding='utf-8') as f:
        f.write(vision_text)
        
    return vision_text

def get_file_hash(file_path):
    """Generates a SHA-256 hash of the file to use as a unique ID."""
    import hashlib

    hasher = hashlib.sha256()
    with open(file_path, 'rb') as f:
        # Read in chunks to avoid memory spikes on massive PDFs
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

def sanitize_for_yaml(data):
    """
    Recursively converts NumPy types to native Python types and tuples to lists 
    to ensure clean, safe YAML serialization.
    """
    if isinstance(data, dict):
        return {k: sanitize_for_yaml(v) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        return [sanitize_for_yaml(v) for v in data]
    # Check for NumPy scalars (they have an .item() method that returns native python types)
    elif hasattr(data, 'item') and callable(data.item):
        try:
            return data.item()
        except Exception:
            return data
    return data