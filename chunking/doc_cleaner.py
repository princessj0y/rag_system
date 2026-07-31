import os
from pathlib import Path

from utils.model_factories import create_model_by_name

from tqdm import tqdm
from utils.my_log import logger

def clean_doc(file_path, is_eng=None):
    """
    Accepts any file path, auto-detects its type, cleans noise, 
    and returns a list of LangChain Document objects using the modern 1.x loader.
    """

    if is_eng is None:
        is_eng = 'EN' in file_path

    # to-do: add case for excel/csv 
    if file_path.lower().endswith(('.png', '.jpg', 'jpeg', '.svg')):
        clean_doc = clean_imageful_doc(file_path)
    else:
        clean_doc = clean_textful_doc(file_path, is_eng)
    
    return clean_doc


def clean_textful_doc(file_path, is_eng):
    import yaml
    from langchain_unstructured import UnstructuredLoader
    from langchain_core.documents import Document
    from utils.documents import preserialize_docs, reconstruct_docs
 
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
            docs = reconstruct_docs(yaml.full_load(f))
    else:
        logger.info(f"Cache miss. Running Unstructured 'hi_res' on {file_path}...")
        loader = UnstructuredLoader(
            file_path=file_path,
            strategy="hi_res",
            # https://tesseract-ocr.github.io/tessdoc/Data-Files-in-different-versions.html
            languages=['eng' if is_eng else 'ita'],
            skip_headers_and_footers=True, # strips page numbers, repetitive document titles at the top of pages, and legal footers
            # rip images and tables and save them
            extract_image_block_types=["Image", "Table"],
            extract_image_block_output_dir=str(image_dir)
        )
        
        docs = loader.load()
        if not docs:
            docs = [ Document(page_content="", metadata={"source": file_path}) ]
        
        with open(cache_file, 'w', encoding='utf-8') as f:
            yaml.dump(preserialize_docs(docs), f, allow_unicode=True, sort_keys=False)
        logger.info(f"Saved parsed output to {cache_file}")

    # Enrich images
    image_docs = [doc for doc in docs if doc.metadata.get("category") in ('Table', 'Image')]
    if image_docs:
        for doc in tqdm(image_docs, desc="Processing extracted images and tables"):
            img_path = doc.metadata.get("image_path")
            if img_path and os.path.exists(img_path):
                category = doc.metadata.get("category")
                try:
                    summary, md = clean_imageful_doc(img_path, 
                                                    is_table=(category == 'Table'),
                                                    is_eng=is_eng,
                                                    cache_path=Path(img_path).with_suffix('.txt'))
                except:
                    logger.exception(f"Failed to extract {category} {img_path}, skipping it...")
                    continue

                # Put it in page_content so it gets embedded and searched
                doc.page_content = summary
                # Tuck a safe copy in the metadata just in case it gets chopped
                doc.metadata["payloads"] = [{
                    "type": f"{category.lower()}_markdown",
                    "raw_content": md
                }]

    # Double-tap the noise
    # Unstructured's internal skip isn't 100% reliable, so we force-drop them here (especially headers)
    ignored_categories = {"Header", "Footer", "PageNumber", "PageBreak"}
    docs = [d for d in docs if d.metadata.get("category") not in ignored_categories]

    # Save post-processed docs just for inspection
    processed_file = cache_dir / "docs-postprocess.yaml"
    with open(processed_file, 'w', encoding='utf-8') as f:
        yaml.dump(preserialize_docs(docs), f, allow_unicode=True, sort_keys=False)

    return docs

def clean_imageful_doc(file_path, is_table=False, is_eng=True, cache_path=None):
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
            vision_text = f.read()
    else:
        logger.info(f"Cache miss. Running vision model on {file_path}...")
        file_extension = file_path.lower().split('.')[-1]
    
        if file_extension == "svg":
            import cairosvg
            png_bytes = cairosvg.svg2png(url=file_path, output_width=1600)
            img_base64 = base64.b64encode(png_bytes).decode("utf-8")
            mime_type = "image/png"
        elif file_extension in ["png", "jpg", "jpeg", "webp"]:
            with open(file_path, "rb") as image_file:
                img_base64 = base64.b64encode(image_file.read()).decode("utf-8")
            mime_type = f"image/{'jpeg' if file_extension == 'jpg' else file_extension}"
        else:
            raise ValueError(f"Unsupported file type for vision processing: {file_extension}")

        answer_llm = create_model_by_name(model="gemma4:31b-cloud")
        
        target_lang = "English" if is_eng else "Italian"
        if is_table:
            prompt = f"""
            You are an expert data extraction assistant. Analyze this table image.
            You MUST structure your response exactly like this:
            
            ---SUMMARY---
            [Write a 1-2 sentence description of what the table shows, including column names and main entities.\
             The summary MUST be written in {target_lang}]
            ---PAYLOAD---
            [Write a perfect, row-by-row Markdown table transcription of all data]
            """
        else:
            prompt = f"""
            You are an expert data extraction assistant. This is an image/infographic filled with statistics.
            Transcribe ALL text, numbers, metrics, and chart data into clean, highly organized Markdown.
            Make sure you explicitly pair metrics with their labels (e.g., 'Passaggi: 1,487 (77% completati)').
            Maintain logical reading order and structure.
            Any descriptive text you generate MUST be written in {target_lang}.
            """

        message = HumanMessage(
            content=[
                {"type" : "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{img_base64}"}}
            ]
        )

        response = answer_llm.invoke([message])
        vision_text = response.content.strip()

        # Save to cache
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, 'w', encoding='utf-8') as f:
            f.write(vision_text)

    if not is_table:
        return  f"Image Description:\n{vision_text}", vision_text
    
    # Robustly split the output using the delimiter
    if "---PAYLOAD---" in vision_text:
        parts = vision_text.split("---PAYLOAD---", 1)

        summary = parts[0].strip()
        payload = parts[1].strip()

        if "---SUMMARY---" in summary:
            summary = summary.split("---SUMMARY---", 1)[1].strip()

        return summary, payload
    
    logger.warning(f"Vision LLM missed the delimiter for {file_path}. Using fallback parsing.")
    return "Visual data extracted.", vision_text.strip()


def get_file_hash(file_path):
    """Generates a SHA-256 hash of the file to use as a unique ID."""
    import hashlib

    hasher = hashlib.sha256()
    with open(file_path, 'rb') as f:
        # Read in chunks to avoid memory spikes on massive PDFs
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()
