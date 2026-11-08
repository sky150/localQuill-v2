"""Initialization for the LLM providers package."""

from providers.base import Provider, Message


def correct_text(provider: Provider, text: str) -> str:
    messages: list[Message] = [
        {"role": "system", "content": "Fix commas, grammar and tense only."},
        {"role": "user", "content": text},
    ]
    return provider.generate(messages, temperature=0.2)