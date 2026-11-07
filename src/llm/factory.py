import logging
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from src.config import (
    get_config, OLLAMA_URL, LMSTUDIO_URL, OPENAI_API_KEY,
    VERBOSE, KEEP_ALIVE, LIMIT_TOKEN_NUMBER, PROVIDER, 
)

logger = logging.getLogger(__name__)


def _keep_alive() -> str | None:
    value = KEEP_ALIVE.lower()
    if value in ("", "false"):
        return None
    if value == "true":
        return "5m"
    return KEEP_ALIVE


def _token_limit(config_limit: int | None) -> int | None:
    limits = [n for n in (config_limit, LIMIT_TOKEN_NUMBER) if n]
    return min(limits) if limits else None


def _drop_none(d: dict) -> dict:
    return {k: v for k, v in d.items() if v is not None}


# One builder per provider. Each maps the shared cfg to that provider's parameter names.

def _build_ollama(cfg) -> BaseChatModel:
    return ChatOllama(**_drop_none({
        "model": cfg.model,
        "base_url": OLLAMA_URL,
        "temperature": cfg.temperature,
        "keep_alive": _keep_alive(),                  # Ollama only
        "num_predict": _token_limit(cfg.max_tokens),  # Ollama name for max tokens
        "num_ctx": 4096,                              # Ollama only
    }))


def _build_lmstudio(cfg) -> BaseChatModel:
    return ChatOpenAI(**_drop_none({
        "model": cfg.model,
        "base_url": LMSTUDIO_URL,                     # e.g. http://localhost:1234/v1
        "api_key": "lm-studio",                       # LM Studio ignores it, SDK requires one
        "temperature": cfg.temperature,
        "max_tokens": _token_limit(cfg.max_tokens),
    }))


def _build_openai(cfg) -> BaseChatModel:
    return ChatOpenAI(**_drop_none({
        "model": cfg.model,
        "api_key": OPENAI_API_KEY,
        "temperature": cfg.temperature,
        "max_tokens": _token_limit(cfg.max_tokens),
    }))


BUILDERS = {
    "ollama": _build_ollama,
    "lmstudio": _build_lmstudio,
    "openai": _build_openai,
}


def get_model(task_name: str, style: str = "formal", cfg=None, provider: str | None = None) -> BaseChatModel:
    """Return the model for a task. cfg and provider are overrides for the eval script."""
    cfg = cfg or get_config(task_name, style)
    provider = provider or PROVIDER

    try:
        build = BUILDERS[provider]
    except KeyError:
        raise ValueError(f"Unknown provider '{provider}'. Use one of: {list(BUILDERS)}")

    log = logger.info if VERBOSE else logger.debug
    log(f"[{task_name}/{style}] provider={provider} model={cfg.model} "
        f"temp={cfg.temperature} collection={cfg.collection} top_k={cfg.top_k}")

    return build(cfg)