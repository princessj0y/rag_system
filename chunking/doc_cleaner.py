import pymupdf4llm

def clean_doc(file_path):
    """
    Accepts a file path, cleans the document, and returns text.
    """
    # to_markdown helps remove PDF 'noise' like headers and footers
    cleaned_text = pymupdf4llm.to_markdown(file_path)
    return cleaned_text