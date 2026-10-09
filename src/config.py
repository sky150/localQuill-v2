# per-task model, temperature, prompt (replaces base_models_dispro1.json)

import os
from dataclasses import dataclass, replace
from dotenv import load_dotenv

load_dotenv()

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
# LMStudio URL
CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma")
DEFAULT_STYLE = "test"


@dataclass(frozen=True)
class TaskConfig:
    model: str
    temperature: float = 0.1
    provider: str = "ollama"          # "ollama", "lmstudio", "openai"
    collection: str | None = None     # Chroma collection; None = no retrieval
    top_k: int = 3
    max_tokens: int | None = None


# Defaults per task
TASKS: dict[str, TaskConfig] = {
    "test": TaskConfig(model="gemma4:e2b", temperature=0.1),   # Test purposes
    "grammar": TaskConfig(model="qwen3.5:4b", collection="style", temperature=0.1),
    "qa": TaskConfig(model="qwen3.5:4b", collection="user_docs", top_k=5, temperature=0.3),
    "feedback": TaskConfig(model="qwen3.5:4b", collection="style", temperature=0.3),
    "autocomplete": TaskConfig(model="gemma4:e2b", temperature=0.5, max_tokens=20),
}

# Which Chroma collection holds the guides for each style
STYLE_COLLECTIONS = {"formal": "formal", "fiction": "fiction"}

# DEV Settings
def _env_bool(name: str) -> bool:
    return os.getenv(name, "false").strip().lower() == "true"

VERBOSE = _env_bool("VERBOSE")
KEEP_ALIVE = os.getenv("KEEP_ALIVE", "").strip()
LIMIT_TOKEN_NUMBER = int(os.getenv("LIMIT_TOKEN_NUMBER") or 0) or None  # None = no cap



def get_config(task: str, style: str = DEFAULT_STYLE) -> TaskConfig:
    """Return the config for a task, with style overrides and env fallbacks applied."""
    config = TASKS[task]
    # Style Overrides (Set models for subtasks. )

    if config.collection == "style":
        config = replace(config, collection=STYLE_COLLECTIONS.get(style, DEFAULT_STYLE))

    # Checks for models like (GRAMMAR_MODEL) to override the model for the grammar task
    if env_model := os.getenv(f"{task.upper()}_MODEL"):
        config = replace(config, model=env_model)
    if env_temperature := os.getenv(f"{task.upper()}_TEMPERATURE"):
        config = replace(config, temperature=float(env_temperature))
        
    return config