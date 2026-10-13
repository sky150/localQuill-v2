import hashlib
import logging
import os

from langchain_chroma import Chroma

from src.config import CHROMA_PATH, UPLOAD_PATH
from src.embedding.embeddings import get_embedding_function
from src.vector_db.chunk_documents import split_documents
from src.vector_db.generate_chroma import add_to_chroma
from src.vector_db.load_documents import SUPPORTED_EXTENSIONS, load_file

logger = logging.getLogger(__name__)

DOCUMENT_KINDS = ("userdocs", "outline")


def doc_id_for(source: str) -> str:
    return hashlib.sha1(os.path.abspath(source).encode("utf-8")).hexdigest()[:12]


def safe_filename(name: str) -> str:
    cleaned = os.path.basename(name.replace("\\", "/")).strip().lstrip(".")
    return cleaned or "document"


def is_supported(filename: str) -> bool:
    return os.path.splitext(filename)[1].lower() in SUPPORTED_EXTENSIONS


def document_path(kind: str, filename: str) -> str:
    return os.path.join(os.path.abspath(UPLOAD_PATH), kind, safe_filename(filename))


def open_collection(kind: str, embedding_function=None) -> Chroma:
    return Chroma(
        persist_directory=os.path.abspath(CHROMA_PATH),
        embedding_function=embedding_function or get_embedding_function(),
        collection_name=kind,
    )


def save_upload(kind: str, filename: str, data: bytes) -> str:
    path = document_path(kind, filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)
    return path


def remove_file(path: str) -> None:
    if os.path.exists(path):
        os.remove(path)


def ingest_document(path: str, kind: str, embedding_function=None) -> dict:
    document = load_file(path, metadata={"kind": kind})
    chunks = split_documents([document])
    db = open_collection(kind, embedding_function)
    db.delete(where={"source": document.metadata["source"]})
    return add_to_chroma(
        chunks, CHROMA_PATH, kind, embedding_function=embedding_function
    )


def list_documents(kind: str | None = None, embedding_function=None) -> list[dict]:
    kinds = [kind] if kind else list(DOCUMENT_KINDS)
    embedding = embedding_function or get_embedding_function()
    documents = []
    for document_kind in kinds:
        rows = open_collection(document_kind, embedding).get(include=["metadatas"])
        grouped: dict[str, int] = {}
        for metadata in rows["metadatas"]:
            source = metadata.get("source")
            if source:
                grouped[source] = grouped.get(source, 0) + 1
        for source, chunks in grouped.items():
            documents.append(
                {
                    "doc_id": doc_id_for(source),
                    "filename": os.path.basename(source),
                    "kind": document_kind,
                    "source": source,
                    "chunks": chunks,
                    "size_bytes": os.path.getsize(source)
                    if os.path.exists(source)
                    else 0,
                }
            )
    return documents


def delete_document(
    doc_id: str, delete_file: bool = False, embedding_function=None
) -> dict:
    embedding = embedding_function or get_embedding_function()
    for kind in DOCUMENT_KINDS:
        db = open_collection(kind, embedding)
        rows = db.get(include=["metadatas"])
        sources = {
            metadata["source"]
            for metadata in rows["metadatas"]
            if metadata.get("source") and doc_id_for(metadata["source"]) == doc_id
        }
        if not sources:
            continue
        deleted_chunks = 0
        file_deleted = False
        for source in sources:
            deleted_chunks += len(db.get(where={"source": source}, include=[])["ids"])
            db.delete(where={"source": source})
            if delete_file and os.path.exists(source):
                os.remove(source)
                file_deleted = True
        return {
            "doc_id": doc_id,
            "kind": kind,
            "deleted_chunks": deleted_chunks,
            "file_deleted": file_deleted,
        }
    return {
        "doc_id": doc_id,
        "kind": None,
        "deleted_chunks": 0,
        "file_deleted": False,
    }
