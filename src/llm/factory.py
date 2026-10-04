from langchain_ollama import OllamaLLM
from src.config import get_config
import logging

logger = logging.getLogger(__name__)

def get_model(task_name: str, style: str = "essay"):
    """Return a model instance based on the task name and style."""
    
    # ToDo: Style is used to determine the model configuration based on the writing style. Add later
    cfg = get_config(task_name, style)
    model = OllamaLLM(
                model=cfg.model,
                base_url="http://127.0.0.1:11434",
                temperature=float(cfg.temperature),
            )
    logger.debug(f"LLM model initialized (Ollama): {cfg.model}")
    return model