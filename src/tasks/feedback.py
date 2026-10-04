from .base import Task

class FeedbackTask(Task):
    name = "feedback"
    
    # Use RAG basis liek model_query.py
    # Use additional custumer RAG
    def build_prompt(self, text, **opts) -> str:
        return text