import os
import logging
from functools import lru_cache
from langchain_chroma import Chroma
from src.config import CHROMA_PATH
from src.guardrails import screen_chunks
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
    """Return a list of (doc, score). Chunks from user-supplied collections are scanned."""
    results = get_db(collection).similarity_search_with_score(query, k=top_k)
    for doc, score in results:
        logger.info(f"[rag] {collection} | score {score:.3f} | {doc.metadata.get('source', '?')}")
    return screen_chunks(results, collection)


# Cross-encoder that reorders style-guide search results: it reads query and passage together,
# so it picks the passage that fits the problem label better than vector distance alone
# (docs/rag-evaluation.md). Downloaded once (~90 MB); without it the vector order is kept.
RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
RERANK_POOL = 30


@lru_cache(maxsize=1)
def get_reranker():
    try:
        from sentence_transformers import CrossEncoder
        return CrossEncoder(RERANK_MODEL)
    except Exception as e:  # missing download, offline, ...
        logger.warning(f"[rag] re-ranker unavailable, using vector order: {e}")
        return None


def retrieve_reranked(collection: str, query: str, top_k: int = 3):
    """The top RERANK_POOL vector results, reordered by the cross-encoder, best top_k."""
    results = retrieve(collection, query, RERANK_POOL)
    reranker = get_reranker()
    if reranker is None or not results:
        return results[:top_k]
    scores = reranker.predict([(query, doc.page_content) for doc, _ in results])
    ranked = [r for r, _ in sorted(zip(results, scores), key=lambda x: -x[1])]
    return ranked[:top_k]


def format_context(results) -> str:
    """Join chunk texts, skipping exact duplicates."""
    seen, chunks = set(), []
    for doc, _score in results:
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            chunks.append(doc.page_content)
    return "\n\n---\n\n".join(chunks)


def format_references(results) -> str:
    """Style-guide chunks, each marked with its source so the model treats it as advice and
    does not review it as the user's text (it did: docs/rag-evaluation.md, leakage)."""
    seen, refs = set(), []
    for doc, _score in results:
        if doc.page_content in seen:
            continue
        seen.add(doc.page_content)
        source = os.path.splitext(os.path.basename(doc.metadata.get("source", "style guide")))[0]
        refs.append(f'REFERENCE {len(refs) + 1}, from "{source}" '
                    f"(advice from a style guide, not the user's text; do not review or quote it):\n"
                    f"{doc.page_content}")
    return "\n\n".join(refs)


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



