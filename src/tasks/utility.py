import re
import os
import logging
from src.embedding import embeddings
from src.guardrails.content_filter import ContentFilterMiddleware
from langchain_chroma import Chroma

logger = logging.getLogger(__name__)

def truncate_by_paragraph(text: str, max_words: int) -> tuple[str, bool]:
    """Keep whole paragraphs until max_words is reached. Returns (text, was_cut)."""
    kept, count = [], 0
    for para in text.split("\n\n"):
        n = len(para.split())
        if kept and count + n > max_words:
            return "\n\n".join(kept), True
        if not kept and n > max_words:
            return " ".join(para.split()[:max_words]), True
        kept.append(para)
        count += n
    return "\n\n".join(kept), False

def clip(text: str, max_chars: int) -> str:
    """Cut at a word boundary. Chunks arrive best-first, so the weakest chunk is cut first."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + " ..."


# Methods moved form old model_query.py
def text_normalization(user_text: str):
    """Normalize text by fixing hyphenated line breaks, normalizing line endings, and collapsing spaces with Regex."""
    # Fix hyphenated line breaks
    text = re.sub(r"(?<=\w)-\n(?=\w)", "", user_text)
    # Replace single newlines with spaces
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    # Normalize multiple paragraph breaks
    text = re.sub(r"\n{2,}", "\n\n", text)
    # Collapse multiple spaces
    text = re.sub(r"[ ]{2,}", " ", text)

    return text.strip()

def use_guardrails(user_text):
    use_guardrails = os.getenv("USE_GUARDRAILS", "true").strip().lower() != "false"
    if use_guardrails:
        cf = ContentFilterMiddleware(logger=logger)
        ok = cf.check_text(user_text)
        if not ok:
            logger.warning("Input blocked by content filter (deterministic check).")
            return "Your input was rejected because it appears to contain unsafe instructions. Please reword and try again."
    return None


def user_text_split(user_text):
# Split at ~5000 characters to Improve quality.
    user_text_split = int(os.getenv("USER_TEXT_SPLIT", "5000"))
    user_text_chunks = embeddings.chunk_user_prompt(user_text, chunk_size=user_text_split, chunk_overlap=int(user_text_split * 0.2))
    return user_text_chunks

def get_db(chroma_path: str, collection_name: str):
    """Initialize and return the ChromaDB collection for the specified style."""
    if not os.path.exists(chroma_path):
        raise FileNotFoundError(f"ChromaDB not found at '{chroma_path}'. ")

    embedding_function = embeddings.get_embedding_function(logger)
    db = Chroma(
        persist_directory=chroma_path,
        embedding_function=embedding_function,
        collection_name=collection_name,
    )

    count = db.get()  # if exists but empty
    if len(count["ids"]) == 0:
        raise ValueError(
            "ChromaDB is empty. " "Run generate_chroma.py first to populate it."
        )

    return db

def similarity_search_eval(db, query: str, top_k: int = 5):
    """
    Plain similarity search for retrieval evaluation.
    Just tests if the embedding model retrieves the right chunks.
    """
    results = db.similarity_search_with_score(query, k=top_k)

    logger.info(f"++ Eval similarity search ++")
    for doc, score in results:
        logger.info(f"Score: {score:.3f} | Source: {doc.metadata.get('source', '?')}")

    return results  # list of (doc, score)
