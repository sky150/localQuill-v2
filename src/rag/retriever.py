import os
import logging
from functools import lru_cache
from langchain_chroma import Chroma
from src.config import CHROMA_PATH
from src.embedding.embeddings import get_embedding_function

logger = logging.getLogger(__name__)

# Cache the Chroma database connections to avoid reopening them multiple times
@lru_cache(maxsize=None)
def get_db(collection: str, chroma_path: str = CHROMA_PATH):
    """Open a Chroma collection once and reuse it."""
    if not os.path.exists(chroma_path):
        raise FileNotFoundError(f"ChromaDB not found at '{chroma_path}'.")
    db = Chroma(
        persist_directory=chroma_path,
        embedding_function=get_embedding_function(logger),
        collection_name=collection,
    )
    if not db.get(limit=1)["ids"]:
        raise ValueError(f"Collection '{collection}' is empty. Run generate_chroma.py.")
    return db


def retrieve(collection: str, query: str, top_k: int = 3):
    """Return a list of (doc, score)."""
    results = get_db(collection).similarity_search_with_score(query, k=top_k)
    for doc, score in results:
        logger.info(f"[rag] {collection} | score {score:.3f} | {doc.metadata.get('source', '?')}")
    return results


def format_context(results) -> str:
    """Join chunk texts, skipping exact duplicates."""
    seen, chunks = set(), []
    for doc, _score in results:
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            chunks.append(doc.page_content)
    return "\n\n---\n\n".join(chunks)


def retrieve_from(self, collection: str, query: str, top_k: int = 3) -> str:
    """Chunks from any collection as one string, or '' if unavailable."""
    try:
        return format_context(retrieve(collection, query, top_k))
    except (FileNotFoundError, ValueError) as e:
        logger.warning(f"[{self.name}] retrieval from '{collection}' skipped: {e}")
        return ""
    
def retrieve_context(self, text: str) -> str:
    """Basic retrieval from one collection from config (grammar, feedback)."""
    if not (self.config.collection and self.retrieval_query):
        return ""
    query = f"{self.retrieval_query}\n\n{text[:1000]}"
    return self.retrieve_from(self.config.collection, query, self.config.top_k)



