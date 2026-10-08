from src.config import get_config
from src.llm.factory import get_model
from .grammar import GrammarTask
from .feedback import FeedbackTask
from .qa import QATask
from .autocomplete import AutocompleteTask
from .test import TestTask

TASK_CLASSES = {
    "grammar": GrammarTask,
    "qa": QATask,
    "feedback": FeedbackTask,
    "autocomplete": AutocompleteTask,
    "test": TestTask,
}

_cache: dict[tuple[str, str], object] = {}
        
def get_task(name: str, style: str = "fiction"):
    """Build a task once per (name, style) and reuse it."""
    key = (name, style)
    if key not in _cache:
        cfg = get_config(name, style)
        _cache[key] = TASK_CLASSES[name](model=get_model(name, style), config=cfg, style=style)
    return _cache[key]