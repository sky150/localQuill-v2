from .base import Task

class TestTask(Task):
    """A simple test task that returns the input text as the prompt."""
    name = "test"

    def build_prompt(self, text, **opts) -> str:
        return text