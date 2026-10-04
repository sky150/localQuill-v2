from langchain_ollama import OllamaLLM
import logging
import os

logger = logging.getLogger(__name__)

def get_model(task_name: str, style: str = "essay"):
    """Return a model instance based on the task name and style."""
    # Model Creation Logic later
    # For example:
    # if task_name == "grammar":
    #     return GrammarModel(style=style)
    # elif task_name == "feedback":
    #     return FeedbackModel(style=style)

    # ToDo: Individual model logic based on task_name and style
    model = OllamaLLM(
                model="gemma4:e2b",
                base_url="http://127.0.0.1:11434",
                temperature=float(os.getenv("TEMPERATURE", "0.1")),
            )
    logger.debug(f"LLM model initialized (Ollama): {'gemma4:e2b'}")
    return model