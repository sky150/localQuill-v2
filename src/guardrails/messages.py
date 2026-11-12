"""Text shown to the user when something is left out or rejected.

Kept apart from policy so wording can change without touching decision logic. Say what
happened and what to do; never include scores or the pattern that matched.
"""

from src.guardrails.types import Source


def message_for(source: Source, name: str | None = None) -> str:
    label = f"'{name}'" if name else "An item"
    if source is Source.USER_TEXT:
        return (
            "Your message was rejected because it appears to contain instructions "
            "aimed at the assistant. Please reword it and try again."
        )
    if source is Source.UPLOAD:
        return (
            f"Some passages in {label} look like instructions to the model. "
            "The file was added, but those passages will be ignored when answering."
        )
    if source is Source.ATTACHMENT:
        return f"{label} was not used: it contains text that looks like instructions to the model."
    if source is Source.RETRIEVED:
        return f"A passage from {label} was left out: it looks like instructions to the model."
    return "Part of the conversation history was ignored because it looks unsafe."
