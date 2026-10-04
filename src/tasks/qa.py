from .base import Task

class QATask(Task):
    name = "qa"
    
    # ToDo: Implement question answering with history
    # ToDo: Integrate RAG for User RAG retrieval
    def build_prompt(self, text, **opts) -> str:
        return text