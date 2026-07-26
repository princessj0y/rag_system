import pandas # fixes pydantic segfault

"""
===============================================================================
HIERARCHICAL LEGAL CHUNKER
===============================================================================
WHY THIS EXISTS:
Legal drafting (tecnica legislativa) follows a strict, predictable hierarchy:
  1. Titolo (Title)   -> Main thematic folder.
  2. Capo (Chapter)   -> Sub-folder. Numbering resets every time a new Titolo starts.
  3. Articolo (Article) -> The atomic, sequential unit of the law. Numbering is 
                           global and DOES NOT reset. This allows lawyers to 
                           cite "Articolo 9" unambiguously.

THE PROBLEM WITH STANDARD CHUNKERS:
Generic recursive or token-based splitters destroy this logic. They blindly chop 
text based on character limits, separating "Articolo 9" from its actual content, 
and losing track of whether we are in "Titolo I" or "Titolo II". 

THE UNSTRUCTURED ELEMENT SOLUTION:
Instead of relying on regex replacements over a flattened string or building an 
AST, this parser streams the pre-parsed Unstructured `Document` objects natively. 
It processes the document element by element to:
  A) Detect structural keywords (Titolo/Capo/Articolo) by scanning short `Title`, 
     `NarrativeText`, or `UncategorizedText` elements.
  B) Maintain a persistent "legal state" tracking exactly which folder we are in.
  C) Buffer all elements (paragraphs, lists, tables) belonging to the active 
     legal node, natively nesting any standard sub-titles beneath it.
  D) Flush the buffer into a unified LangChain chunk whenever a new legal 
     boundary is crossed, flawlessly merging and preserving all rich metadata 
     (like HTML tables) in the process.
===============================================================================
"""

def run_hierarchical_legal_chunking(docs, is_eng=False):
    import re
    from langchain_core.documents import Document
    from utils.documents import merge_metadata, pick_content_separator

    # State tracking 
    current_legal = {"L1": None, "L2": None, "L3": None} # L1=Titolo, L2=Capo, L3=Articolo
    current_title = None # Tracks standard Unstructured Titles
    
    # Dynamic language keywords
    legal_pattern = (r"^(TITLE|CHAPTER|ARTICLE|ART\.)\s+([A-Za-z0-9\-]+)" if is_eng 
                     else r"^(TITOLO|CAPO|ARTICOLO|ART\.)\s+([A-Za-z0-9\-]+)")
    l1_keys = ("TITLE",) if is_eng else ("TITOLO",)
    l2_keys = ("CHAPTER",) if is_eng else ("CAPO",)
    l3_keys = ("ARTICLE", "ART.") if is_eng else ("ARTICOLO", "ART.")

    def get_current_path():
        path = []
        if current_legal["L1"]: path.append(current_legal["L1"])
        if current_legal["L2"]: path.append(current_legal["L2"])
        if current_legal["L3"]: path.append(current_legal["L3"])
        if current_title: path.append(current_title)
        return path

    chunks = []
    current_buffer = []

    def save_chunk():
        """Combines buffered docs into a single chunk and merges their metadata."""
        if not current_buffer:
            return
            
        # Context-aware text reconstruction for the chunk
        text_parts = []
        for i, d in enumerate(current_buffer):
            category = d.metadata.get("category", "")
            next_cat = current_buffer[i+1].metadata.get("category", "") if i + 1 < len(current_buffer) else ""

            text_parts.append(d.page_content)
            text_parts.append(pick_content_separator(category, next_cat))
            
        content = "".join(text_parts).strip()
        
        if content:
            # Safely merge Unstructured metadata (tables, images, etc.)
            unique_metas = list({id(d.metadata): d.metadata for d in current_buffer}.values())
            merged_meta = merge_metadata(unique_metas)
            
            # Inject our legal hierarchy tracking
            merged_meta["heading_path"] = get_current_path()
            merged_meta["level"] = len(get_current_path())
            
            chunks.append(Document(page_content=content, metadata=merged_meta))
            
        current_buffer.clear()
    
    
    def process_legal_node(node_type, clean_text):
        """Helper to update the legal hierarchy state without duplication."""
        nonlocal current_title

        if node_type in l1_keys:
            current_legal["L1"] = clean_text
            current_legal["L2"] = None
            current_legal["L3"] = None
        elif node_type in l2_keys:
            current_legal["L2"] = clean_text
            current_legal["L3"] = None
        elif node_type in l3_keys:
            current_legal["L3"] = clean_text
                
        # Reset generic title anytime a hard legal boundary is crossed
        current_title = None 

    # Iterate through the Unstructured Elements
    for doc in docs:
        category = doc.metadata.get("category", "")
        raw_text = doc.page_content.strip()
        
        # Check for legal headers in likely categories (skip long paragraphs)
        if category in ("Title", "NarrativeText", "UncategorizedText") and len(raw_text) < 100:
            match = re.match(legal_pattern, raw_text, re.IGNORECASE)
            
            # Ensure it's a standalone header, not a paragraph starting with "Article 9..."
            if match and "\n" not in raw_text:
                # A new legal section means we must close out the previous chunk
                save_chunk()
                
                node_type = match.group(1).upper()
                process_legal_node(node_type, raw_text)
                
                # Add the legal header itself to the fresh buffer
                current_buffer.append(doc)
                continue

        # If it's NOT a legal node but still a generic Unstructured Title
        if category == "Title":
            save_chunk() # Titles logically start a new structural section
            current_title = raw_text
            current_buffer.append(doc)
            continue

        # Standard content (NarrativeText, ListItems, Tables, Images)
        current_buffer.append(doc)

    # Final flush for the last section of the document
    save_chunk()
    return chunks

if __name__ == "__main__":
    import yaml
    from pathlib import Path
    from .doc_cleaner import clean_doc
    from utils.documents import preserialize_docs, flatten_metadata_for_chroma

    #file_to_analyze = "./test/CELEX_32006L0054_EN_TXT.pdf"
    file_to_analyze = "./test/CELEX_32006L0054_IT_TXT.pdf"
    is_eng = 'EN' in file_to_analyze

    raw_text = clean_doc(file_to_analyze, is_eng)

    chunks = run_hierarchical_legal_chunking(raw_text, is_eng)
    
    Path("tmp").mkdir(parents=True, exist_ok=True)    
    with open("tmp/chunks-hierarchical-legal.yaml", 'w', encoding='utf-8') as f:
        yaml.dump(preserialize_docs(chunks), f, allow_unicode=True, sort_keys=False)
        
    for chunk in chunks:
        chunk.metadata = flatten_metadata_for_chroma(chunk.metadata)
    with open("tmp/chunks-hierarchical-legal-flattened.yaml", 'w', encoding='utf-8') as f:
        yaml.dump(preserialize_docs(chunks), f, allow_unicode=True, sort_keys=False)