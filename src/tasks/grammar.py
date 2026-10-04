from .base import Task
import logging

logger = logging.getLogger(__name__)

class GrammarTask(Task):
    name = "grammar"
    
    # ToDo: Implement grammar correction using RAG basis like model_query.py
    def build_prompt(self, text, **opts) -> str:
        return text