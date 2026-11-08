"""OpenAI-compatible LLM provider implementation for the localQuill project.

Implements OpenAI and LM Studio providers."""

from collections.abc import Iterator
from openai import OpenAI
from .base import Provider, Message


class OpenAICompatibleProvider(Provider):
    """Shared logic for any server speaking the OpenAI API."""

    def __init__(self, model: str, base_url: str | None, api_key: str) -> None:
        super().__init__(model)
        self.client = OpenAI(base_url=base_url, api_key=api_key)

    def generate(self, messages: list[Message], temperature: float = 0.7) -> str:
        r = self.client.chat.completions.create(
            model=self.model, messages=messages, temperature=temperature
        )
        return r.choices[0].message.content or ""

    def stream(self, messages: list[Message], temperature: float = 0.7) -> Iterator[str]:
        s = self.client.chat.completions.create(
            model=self.model, messages=messages, temperature=temperature, stream=True
        )
        for chunk in s:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def list_models(self) -> list[str]:
        return [m.id for m in self.client.models.list()]


class OpenAIProvider(OpenAICompatibleProvider):
    name = "openai"

    def __init__(self, model: str, api_key: str) -> None:
        super().__init__(model, base_url=None, api_key=api_key)


class LMStudioProvider(OpenAICompatibleProvider):
    name = "lmstudio"

    def __init__(self, model: str, base_url: str = "http://localhost:1234/v1") -> None:
        super().__init__(model, base_url=base_url, api_key="lm-studio")