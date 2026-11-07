"""Base classes and interfaces for LLM providers."""

from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import TypedDict


class Message(TypedDict):
    role: str      # "system" | "user" | "assistant"
    content: str


class Provider(ABC):
    """Contract every LLM provider must follow."""

    name: str  # set by each subclass

    def __init__(self, model: str) -> None:
        self.model = model

    @abstractmethod
    def generate(self, messages: list[Message], temperature: float = 0.7) -> str:
        """Return the full completion as a string."""

    @abstractmethod
    def stream(self, messages: list[Message], temperature: float = 0.7) -> Iterator[str]:
        """Yield the completion in text chunks."""

    @abstractmethod
    def list_models(self) -> list[str]:
        """Return available model names."""