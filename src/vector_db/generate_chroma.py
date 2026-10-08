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

logger = logging.getLogger(__name__)


def add_to_chroma(chunks: list[Document], chroma_path="./chroma", collection_name="default"):
    if not chunks:
        logger.warning(
            f"No chunks to add for collection '{collection_name}'. Skipping."
        )
        print(f"No chunks to add for collection '{collection_name}'. Skipping.")
        return

    abs_chroma_path = os.path.abspath(chroma_path)
    db = Chroma.from_documents(
        documents=chunks,
        embedding=get_embedding_function(),
        persist_directory=abs_chroma_path,
        collection_name=collection_name,
    )
    logger.info(f"Successfully saved {len(chunks)} chunks to {chroma_path}")
    logger.info(f"Collection '{collection_name}' now has {len(db.get()['ids'])} documents.")
    print(f"Successfully saved {len(chunks)} chunks to {chroma_path}")
    print(f"Collection '{collection_name}' now has {len(db.get()['ids'])} documents.")

    # Add or Update the documents.
    existing_items = db.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"])
    logger.info(f"Number of existing documents in DB: {len(existing_ids)}")

    # Only add documents that don't exist in the DB.
    new_chunks = []
    chunks_with_ids = calculate_chunk_ids(chunks)
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        logger.info(f"Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(
            new_chunks, ids=new_chunk_ids
        ) 
    else:
        logger.info("No new documents to add")


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
        # Call all the relevant methods
        documents = load_documents(data_path)
        chunks = split_documents(documents)
        add_to_chroma(chunks, chroma_path, collection_name)


if __name__ == "__main__":
    main()
