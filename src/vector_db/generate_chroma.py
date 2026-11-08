import os
import argparse
import logging
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_chroma import Chroma
from src.embedding.embeddings import get_embedding_function
from src.vector_db.load_documents import load_documents
from src.vector_db.chunk_documents import split_documents, calculate_chunk_ids
from src.vector_db.reset_db import reset_db
from src.vector_db.load_documents import iter_documents

logger = logging.getLogger(__name__)



BATCH_SIZE = int(os.getenv("INGEST_BATCH_SIZE", "32"))


def _open_db(chroma_path, collection_name, embedding_function=None):
    return Chroma(
        persist_directory=os.path.abspath(chroma_path),
        embedding_function=embedding_function or get_embedding_function(),
        collection_name=collection_name,
    )


def _add_new_chunks(db, chunks, existing_ids, batch_size=BATCH_SIZE) -> int:
    """Embed and store only unseen chunks, a small batch at a time."""
    chunks_with_ids = calculate_chunk_ids(chunks)
    new_chunks = [c for c in chunks_with_ids if c.metadata["id"] not in existing_ids]
    for i in range(0, len(new_chunks), batch_size):
        batch = new_chunks[i : i + batch_size]
        ids = [c.metadata["id"] for c in batch]
        db.add_documents(batch, ids=ids)
        existing_ids.update(ids)
    return len(new_chunks)


def add_to_chroma(chunks, chroma_path="./chroma", collection_name="default", embedding_function=None):
    if not chunks:
        logger.warning(f"No chunks to add for collection '{collection_name}'. Skipping.")
        return {"added": 0, "skipped": 0, "total": 0}
    db = _open_db(chroma_path, collection_name, embedding_function)
    existing_ids = set(db.get(include=[])["ids"])
    added = _add_new_chunks(db, chunks, existing_ids)
    result = {"added": added, "skipped": len(chunks) - added, "total": len(existing_ids)}
    print(f"Collection '{collection_name}': {result}")
    return result


def ingest_directory(data_path, chroma_path, collection_name):
    """One file at a time: load, split, embed in batches, free memory."""
    db = _open_db(chroma_path, collection_name)
    existing_ids = set(db.get(include=[])["ids"])
    added = total_chunks = 0
    for doc in iter_documents(data_path):
        chunks = split_documents([doc])
        n_new = _add_new_chunks(db, chunks, existing_ids)
        added += n_new
        total_chunks += len(chunks)
        print(f"{doc.metadata['filename']}: {len(chunks)} chunks, {n_new} new")
    print(f"Collection '{collection_name}': {added} added, "
          f"{total_chunks - added} skipped, {len(existing_ids)} total.")



def main():
    # Hyperparameters are loaded from .env file
    load_dotenv()
    data_path = os.getenv("DATA_PATH", "./data/styles/formal")
    chroma_path = os.getenv("CHROMA_PATH", "./chroma")
    collection_name = os.getenv("COLLECTION_NAME", "formal")

    # Check if the database should be cleared (using the --reset flag).
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()
    if args.reset:
        reset_db(chroma_path, logger)
    else:
        ingest_directory(data_path, chroma_path, collection_name)

if __name__ == "__main__":
    main()
