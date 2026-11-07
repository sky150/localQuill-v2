"""Ollama LLM provider implementation for the localQuill project."""

from collections.abc import Iterator
import ollama
from .base import Provider, Message


class OllamaProvider(Provider):
    name = "ollama"

    def __init__(self, model: str, host: str = "http://localhost:11434") -> None:
        super().__init__(model)
        self.client = ollama.Client(host=host)

    def generate(self, messages: list[Message], temperature: float = 0.7) -> str:
        r = self.client.chat(
            model=self.model, messages=messages, options={"temperature": temperature}
        )
        return r["message"]["content"]

    def stream(self, messages: list[Message], temperature: float = 0.7) -> Iterator[str]:
        for chunk in self.client.chat(
            model=self.model, messages=messages,
            options={"temperature": temperature}, stream=True,
        ):
            yield chunk["message"]["content"]

    def list_models(self) -> list[str]:
        return [m["model"] for m in self.client.list()["models"]]