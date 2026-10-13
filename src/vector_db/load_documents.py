import logging
import os

from langchain_core.documents import Document
from markitdown import FileConversionException, MarkItDown, UnsupportedFormatException

logger = logging.getLogger(__name__)
logging.getLogger("pdfminer").setLevel(logging.ERROR)

SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".pptx",
    ".xlsx",
    ".xls",
    ".txt",
    ".md",
    ".markdown",
    ".csv",
    ".json",
    ".xml",
    ".html",
    ".htm",
    ".epub",
}

_markitdown = MarkItDown(enable_plugins=False)


def load_file(path: str, metadata: dict | None = None) -> Document:
    """Convert a single file to Markdown and wrap it in a LangChain Document."""
    result = _markitdown.convert_local(path)
    meta = {
        "source": os.path.abspath(path),
        "filename": os.path.basename(path),
    }
    if metadata:
        meta.update(metadata)
    return Document(page_content=result.markdown, metadata=meta)


def load_documents(data_path: str, metadata: dict | None = None) -> list[Document]:
    """Convert every supported file in a directory to Markdown Documents."""
    documents = []
    for name in sorted(os.listdir(data_path)):
        path = os.path.join(data_path, name)
        if not os.path.isfile(path):
            continue
        if os.path.splitext(name)[1].lower() not in SUPPORTED_EXTENSIONS:
            logger.warning(f"Skipping unsupported file: {path}")
            continue
        try:
            documents.append(load_file(path, metadata))
        except (UnsupportedFormatException, FileConversionException) as e:
            logger.error(f"Skipping {path}: {e}")
    return documents
