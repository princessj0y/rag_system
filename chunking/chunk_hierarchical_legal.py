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

THE PROBLEM WITH PDFs & LANGCHAIN:
PDF-to-Markdown extractors destroy this logic. They often flatten the hierarchy 
by assigning all headings to the same level (e.g., `##`), which causes standard 
splitters (like LangChain) to overwrite parent folders instead of nesting them. 
Furthermore, PDF extractors often fail to recognize legal nodes entirely, leaving 
them as isolated bold/italic paragraphs (e.g., `**Articolo 9**` on its own line).

THE AST SOLUTION:
Instead of relying on regex replacements or generic Markdown splitters, this 
parser uses an Abstract Syntax Tree (AST) via `markdown-it-py`. It scans the 
token stream to:
  A) Detect structural keywords (Titolo/Capo/Articolo) whether they are formatted 
     as Markdown Headings (#) OR as isolated formatted paragraphs.
  B) Maintain a persistent "legal state" tracking exactly which folder we are in.
  C) Re-nest any generic Markdown sub-headings natively beneath the active Articolo.
  D) Slice the original Markdown string using token line boundaries, perfectly 
     preserving tables, bullet lists, and original text spacing in the chunk.
===============================================================================
"""

def run_hierarchical_legal_chunking(md_text, is_eng=False):
    import re
    from markdown_it import MarkdownIt
    from langchain_core.documents import Document

    md = MarkdownIt()
    tokens = md.parse(md_text)
    lines = md_text.split("\n")
    
    chunks = []
    
    # State tracking 
    current_legal = {"L1": None, "L2": None, "L3": None} # L1=Titolo, L2=Capo, L3=Articolo
    current_md_headings = {} # Tracks standard markdown sub-headings by level
    
    last_path = []
    last_boundary_line = 0

    # Dynamic language keywords
    legal_pattern = (r"^(TITLE|CHAPTER|ARTICLE|ART\.)\s+([A-Za-z0-9\-]+)" if is_eng 
                     else r"^(TITOLO|CAPO|ARTICOLO|ART\.)\s+([A-Za-z0-9\-]+)")
    l1_keys = ("TITLE",) if is_eng else ("TITOLO",)
    l2_keys = ("CHAPTER",) if is_eng else ("CAPO",)
    l3_keys = ("ARTICLE", "ART.") if is_eng else ("ARTICOLO", "ART.")

    def update_last_path():
        path = []
        if current_legal["L1"]: path.append(current_legal["L1"])
        if current_legal["L2"]: path.append(current_legal["L2"])
        if current_legal["L3"]: path.append(current_legal["L3"])
        for lvl in sorted(current_md_headings.keys()):
            path.append(current_md_headings[lvl])
        return path

    def save_chunk(end_line):
        if last_boundary_line < end_line:
            content = "\n".join(lines[last_boundary_line:end_line]).strip()
            if content:
                chunks.append(Document(
                    page_content=content,
                    metadata={
                        "heading_path": list(last_path),
                        "level": len(last_path)
                    }
                ))

    def process_legal_node(node_type, clean_text):
        """Helper to update the legal hierarchy state without duplication."""
        if node_type in l1_keys:
            current_legal["L1"] = clean_text
            current_legal["L2"] = None
            current_legal["L3"] = None
        elif node_type in l2_keys:
            current_legal["L2"] = clean_text
            current_legal["L3"] = None
        elif node_type in l3_keys:
            current_legal["L3"] = clean_text
        
        # Reset standard markdown sub-headings anytime a legal node is hit
        current_md_headings.clear()

    for i, token in enumerate(tokens):
        
        # 1. Handle Actual Markdown Headings (#, ##, ###)
        if token.type == "heading_open":
            start_line, end_line = token.map
            save_chunk(start_line)
            
            inline_token = tokens[i+1]
            raw_clean_text = inline_token.content.strip("*_ ")
            level = int(token.tag[1])
            
            match = re.match(legal_pattern, raw_clean_text, re.IGNORECASE)
            
            if match:
                node_type = match.group(1).upper()
                process_legal_node(node_type, raw_clean_text)
            else:
                # Standard markdown heading logic (Sub-MD structure)
                keys_to_remove = [k for k in current_md_headings.keys() if k >= level]
                for k in keys_to_remove: del current_md_headings[k]
                current_md_headings[level] = raw_clean_text
                
            last_path = update_last_path()
            last_boundary_line = end_line

        # 2. Handle Isolated Paragraphs (PDF quirks: **Articolo 9** on its own line)
        elif token.type == "paragraph_open":
            start_line, end_line = token.map
            inline_token = tokens[i+1]
            raw_clean_text = inline_token.content.strip("*_ ")
            
            # Isolated legal node heuristic: Matches prefix, < 100 chars, single line
            match = re.match(f"{legal_pattern}(.*)$", raw_clean_text, re.IGNORECASE)
            
            if match and len(raw_clean_text) < 100 and "\n" not in inline_token.content:
                save_chunk(start_line)
                
                node_type = match.group(1).upper()
                process_legal_node(node_type, raw_clean_text)
                    
                last_path = update_last_path()
                last_boundary_line = end_line

    save_chunk(len(lines))
    return chunks

if __name__ == "__main__":
    from .doc_cleaner import clean_doc

    #file_to_analyze = "./test/CELEX_32006L0054_EN_TXT.pdf"
    file_to_analyze = "./test/CELEX_32006L0054_IT_TXT.pdf"
    raw_text = clean_doc(file_to_analyze)

    run_hierarchical_legal_chunking(raw_text, 'EN' in file_to_analyze)