"""PDF text extraction utilities."""

import re


def extract_text_from_pdf(file_path: str) -> list[str]:
    """
    Extract text from a PDF file, one string per page.

    Args:
        file_path: Path to the PDF file.

    Returns:
        List of page texts (one entry per page).
    """
    from PyPDF2 import PdfReader

    pages = []

    def visitor_text(text, cm, tm, fontDict, fontSize):
        y = tm[5]
        if y > 20 and text:
            text = text.replace("\n", " ")
            text = re.sub(r"\[\d+\]|➢|•", "", text)
            parts.append(text)

    reader = PdfReader(file_path)
    for page in reader.pages:
        parts = []
        page.extract_text(visitor_text=visitor_text)
        pages.append(re.sub(r"\s+", " ", " ".join(parts)).strip())

    return pages


def get_pdf_documents(file_path: str) -> list:
    """
    Load a PDF as a list of LangChain Documents (one per page).

    Args:
        file_path: Path to the PDF file.

    Returns:
        List of Document objects with page_content and metadata.
    """
    from langchain_core.documents import Document

    documents = []
    for page_num, text in enumerate(extract_text_from_pdf(file_path)):
        documents.append(
            Document(
                page_content=text,
                metadata={"page": page_num + 1, "id": f"p{page_num + 1}", "source": file_path},
            )
        )
    return documents
