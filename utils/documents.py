import yaml

# Tell PyYAML to use block scalars (|) for multiline strings for ultimate readability
def str_presenter(dumper, data):
    if len(data.splitlines()) > 1:
        return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='|')
    return dumper.represent_scalar('tag:yaml.org,2002:str', data)
    
yaml.add_representer(str, str_presenter)

def preserialize_docs(docs):
    # Langchain documents use Pydantic under the hood. 
    # model_dump() works for v2, dict() for v1.
    # Sanitize the dictionary to remove NumPy types and Tuples
    return _sanitize_for_yaml([
        doc.model_dump() if hasattr(doc, 'model_dump') else doc.dict() 
        for doc in docs
    ])

def _sanitize_for_yaml(data):
    """
    Recursively converts NumPy types to native Python types and tuples to lists 
    to ensure clean, safe YAML serialization.
    """
    if isinstance(data, dict):
        return {k: _sanitize_for_yaml(v) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        return [_sanitize_for_yaml(v) for v in data]
    # Check for NumPy scalars (they have an .item() method that returns native python types)
    elif hasattr(data, 'item') and callable(data.item):
        try:
            return data.item()
        except Exception:
            return data
    return data

def reconstruct_docs(cached_data):
    from langchain_core.documents import Document
    return [Document(**doc_dict) for doc_dict in cached_data]

def make_chunking_document_aware(chunking_function):
    """
    A universal wrapper that takes a raw-text chunking function and upgrades it 
    to accept LangChain Documents, tracking and merging metadata automatically.
    """
    """
    A universal wrapper that accepts a raw-text chunking function returning EITHER lists of strings 
    OR lists of LangChain Documents, upgrading it  to accept LangChain Documents, tracking and merging 
    metadata automatically and with any metadata the chunking function itself generated.
    """
    from langchain_core.documents import Document

    def wrapper(docs, *args, **kwargs):
        # Flatten the text and record spans
        full_text = ""
        doc_spans = [] 
    
        for i, doc in enumerate(docs):
            start_idx = len(full_text)
            
            category = doc.metadata.get("category", "")
            next_category = docs[i+1].metadata.get("category", "") if i + 1 < len(docs) else ""
            
            full_text += doc.page_content + pick_content_separator(category, next_category)
            end_idx = len(full_text)
            
            doc_spans.append((start_idx, end_idx, doc.metadata))

        # Run the naive, off-the-shelf chunker
        # We pass *args and **kwargs so er can still pass chunk_size, is_eng, etc.
        raw_chunks = chunking_function(full_text, *args, **kwargs)
            
        final_docs = []
        search_start = 0

        # Re-attribute metadata based on character overlap
        for chunk_item in raw_chunks:
            # Handle both strings and Document objects gracefully
            if isinstance(chunk_item, str):
                chunk_text = chunk_item
                chunker_meta = {}
            else:
                chunk_text = chunk_item.page_content
                chunker_meta = chunk_item.metadata

            # Find positional overlap
            chunk_start, chunk_end = find_flexible_bounds(chunk_text, full_text, search_start)
            search_start = chunk_end

            # Find overlapping documents
            overlapping_metas = []
            # Gather the original metadata spanning this text
            for span_start, span_end, meta in doc_spans:
                if chunk_start < span_end and chunk_end > span_start:
                    overlapping_metas.append(meta)
            # Gather the metadata the chunking function generated
            if chunker_meta:
                overlapping_metas.append(chunker_meta)

            # Deduplicate (O(N) identity check) and Merge
            unique_metas = list({id(m): m for m in overlapping_metas}.values())
            merged_meta = merge_metadata(unique_metas)

            final_docs.append(Document(page_content=chunk_text, metadata=merged_meta))
            
        return final_docs

    import re

    def find_flexible_bounds(chunk_text, full_text, search_start=0):
        """
        Finds the start and end index of a chunk in the full_text, 
        even if the chunker modified whitespace, newlines, or stripped edges.
        """
        # Try the fast, exact match first
        exact_idx = full_text.find(chunk_text, search_start)
        if exact_idx != -1:
            return exact_idx, exact_idx + len(chunk_text)

        # Fallback: Whitespace-agnostic anchor matching
        tokens = chunk_text.split()
        if not tokens:
            return search_start, search_start

        # Grab up to the first 7 words and last 7 words to create unique anchors
        start_tokens = tokens[:7]
        end_tokens = tokens[-7:]

        # \s* matches ANY whitespace (newlines, tabs, spaces, or nothing)
        start_pattern = r'\s*'.join(re.escape(t) for t in start_tokens)
        end_pattern = r'\s*'.join(re.escape(t) for t in end_tokens)

        # Find where the chunk actually begins in the original text
        start_match = re.search(start_pattern, full_text[search_start:])
        chunk_start = search_start + start_match.start() if start_match else search_start

        # Find where the chunk ends, searching from the start point
        end_match = re.search(end_pattern, full_text[chunk_start:])
        chunk_end = chunk_start + end_match.end() if end_match else chunk_start + len(chunk_text)

        return chunk_start, chunk_end

    return wrapper

def pick_content_separator(category, next_category):
    # --- Context-Aware Whitespace Logic ---
    if category == "ListItem" and next_category == "ListItem":
        # Keep lists tightly packed
        return "\n" 
    elif category == "FigureCaption" and next_category in ("Image", "Table"):
        # Bind captions tightly to the element they precede
        return "\n"            
    elif category in ("Image", "Table") and next_category == "FigureCaption":
        # Bind captions tightly to the element they follow
        return  "\n" 
    elif category == "CodeSnippet" and next_category == "CodeSnippet":
        # OCR often splits code blocks; glue them back together
        return "\n" 
    elif category in ("Address", "EmailAddress") and next_category in ("Address", "EmailAddress"):
        # Keep contact blocks together
        return "\n"
    # Titles, NarrativeText, Formulas, and default paragraph breaks
    # \n\n allows standard chunkers to recognize logical breaks
    return "\n\n"

def merge_metadata(metadata_list, blacklist={"detection_class_prob", "coordinates", "category"}):
    merged = {}
    
    for meta in metadata_list:
        for k, v in meta.items():
            if k in blacklist or v is None:
                continue
            
            if k not in merged:
                merged[k] = v
            else:
                current_val = merged[k]
                if current_val == v:
                    continue
                
                # Convert both to sets for easy union (handling strings vs lists)
                current_set = set(current_val) if isinstance(current_val, (list, tuple)) else {current_val}
                new_set = set(v) if isinstance(v, (list, tuple)) else {v}
                
                union_set = current_set.union(new_set)
                
                # If it's just one item, keep it as a primitive, otherwise list
                merged[k] = list(union_set) if len(union_set) > 1 else list(union_set)[0]
                
    return merged

def flatten_metadata_for_chroma(metadata, parent_key=''):
    items = []
    for k, v in metadata.items():
        # Construct the key path
        new_key = f"{parent_key}.{k}" if parent_key else k
        
        if isinstance(v, dict):
            items.extend(flatten_metadata_for_chroma(v, new_key).items())
        elif isinstance(v, list):
            for i, item in enumerate(v):
                list_key = f"{new_key}[{i}]"
                if isinstance(item, dict):
                    items.extend(flatten_metadata_for_chroma(item, list_key).items())
                else:
                    items.append((list_key, item))
        else:
            # Chroma primitives check
            if isinstance(v, (str, int, float, bool)):
                items.append((new_key, v))
            elif v is not None:
                # Force anything weird (like a set) into a string
                items.append((new_key, str(v)))
                
    return dict(items)