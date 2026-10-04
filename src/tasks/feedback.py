from .base import Task
import logging

logger = logging.getLogger(__name__)

class FeedbackTask(Task):
    name = "feedback"
    
    # Use RAG basis liek model_query.py
    # Use additional custumer RAG
    def build_prompt(self, text, **opts) -> str:
        return text