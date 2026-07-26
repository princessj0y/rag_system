import pandas # fixes pydantic segfault

def run_hierarchical_chunking(md_text, is_eng=False):
    from langchain_text_splitters import MarkdownHeaderTextSplitter

    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
        ("####", "Header 4"),
        ("#####", "Header 5"),
        ("######", "Header 6"),
    ]

    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on)
    chunks = markdown_splitter.split_text(md_text)
    return chunks

if __name__ == "__main__":
    from .doc_cleaner import clean_doc

    #file_to_analyze = "./test/CELEX_32006L0054_EN_TXT.pdf"
    #file_to_analyze = "./test/CELEX_32006L0054_IT_TXT.pdf"
    file_to_analyze = "./test/Strategia_italiana_per_l_Intelligenza_artificiale_2024-2026.pdf"
    raw_text = clean_doc(file_to_analyze)

    run_hierarchical_chunking(raw_text, 'EN' in file_to_analyze)