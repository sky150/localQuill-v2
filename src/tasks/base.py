import logging
import time

logger = logging.getLogger(__name__)

class Task:
    """Base class for all tasks."""
    name: str

    def __init__(self, model, retriever=None):
        self.model = model
        self.retriever = retriever

    def build_prompt(self, text, **opts) -> str:
        """Build the prompt to be sent to the model. Must be implemented by subclasses."""
        raise NotImplementedError(f"{type(self).__name__} must implement build_prompt")

    def run(self, text, **opts) -> str:
        """Run the task by building the prompt and invoking the model."""
        start_time = time.time()
        response = self.model.invoke(self.build_prompt(text, **opts))
        end_time = time.time()
        logger.debug(f"Task {self.name} ran in {end_time - start_time:.2f} seconds")
        # Ollama: str, OpenAI: .content
        return getattr(response, "content", response) + f" (ran in {end_time - start_time:.2f} seconds)"  
        