class Task:
    name: str
    def __init__(self, model, retriever=None):
        self.model = model
        self.retriever = retriever

    def build_prompt(self, text, **opts) -> str: ...
    def run(self, text, **opts) -> str:
        return self.model.invoke(self.build_prompt(text, **opts))