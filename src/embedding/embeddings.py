import os
from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter 


load_dotenv()


def get_embedding_function(logger=None):
    provider = os.getenv("EMBEDDING_PROVIDER", "huggingface")
    if logger:
        logger.info(f"Embedding provider: {provider}")
        logger.info(f"Embedding model: {os.getenv('EMBEDDING_MODEL')}")

    if provider == "ollama":
        return OllamaEmbeddings(
            model=os.getenv("EMBEDDING_MODEL", "nomic-embed-text"),
            base_url=os.getenv("OLLAMA_URL", "http://127.0.0.1:11434"),
        )

    elif provider == "huggingface":
        return HuggingFaceEmbeddings(
            model_name=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        )

    else:
        raise ValueError(f"Unknown embedding provider: {provider}")
    




def chunk_user_prompt(text: str, chunk_size: int, chunk_overlap: int):
    """Splits the user prompt into chunks of a specified size based on words with a specified overlap."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". " " ", ""]
    )

    return splitter.split_text(text)
