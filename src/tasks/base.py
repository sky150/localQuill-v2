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
        response = self.model.invoke(self.build_prompt(text, **opts))
        return getattr(response, "content", response)  # Ollama: str, OpenAI: .content