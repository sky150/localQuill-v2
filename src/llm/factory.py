from langchain_ollama import OllamaLLM
from src.config import get_config, OLLAMA_URL, VERBOSE, KEEP_ALIVE, LIMIT_TOKEN_NUMBER
import logging

logger = logging.getLogger(__name__)


def _keep_alive() -> str | None:
    """Map the KEEP_ALIVE env value to what Ollama accepts."""
    value = KEEP_ALIVE.lower()
    if value in ("", "false"):
        return None
    if value == "true":
        return "10m"
    return KEEP_ALIVE  # e.g. "10m"


def _token_limit(config_limit: int | None) -> int | None:
    """Use the lower of the task limit and the dev limit."""
    limits = [n for n in (config_limit, LIMIT_TOKEN_NUMBER) if n]
    return min(limits) if limits else None


def get_model(task_name: str, style: str = "formal"):
    """Return a model instance based on the task name and style."""
    cfg = get_config(task_name, style)

    kwargs = {
        "model": cfg.model,
        "base_url": OLLAMA_URL,
        "temperature": cfg.temperature,
        "keep_alive": _keep_alive(),                    # Model stays loaded in Ollama
        "num_predict": _token_limit(cfg.max_tokens),    # Token limit for the model
        # ToDo: Move to Python config per model
        "num_ctx": 4096,                                # Context window size for the model
    }
    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    log = logger.info if VERBOSE else logger.debug
    log(f"[{task_name}/{style}] settings: {kwargs} | collection={cfg.collection} top_k={cfg.top_k}")

    return OllamaLLM(**kwargs)