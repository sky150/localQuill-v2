from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum


class Source(StrEnum):
    """Where a piece of text entered the system. Drives trust level and policy."""

    USER_TEXT = "user_text"      # chat message typed by the user
    ATTACHMENT = "attachment"    # workspace file attached to a chat request
    RETRIEVED = "retrieved"      # chunk returned from Chroma (userdocs / outline)
    HISTORY = "history"          # earlier turns sent back by the client
    UPLOAD = "upload"            # file being indexed into Chroma


class Action(StrEnum):
    ALLOW = "allow"
    DROP = "drop"    # leave this item out, carry on with the request
    BLOCK = "block"  # reject the whole request


@dataclass(frozen=True)
class GuardInput:
    source: Source
    text: str
    name: str | None = None   # file name, for messages shown to the user


@dataclass(frozen=True)
class Finding:
    """One layer's reason for concern. Layers return these; policy decides what to do."""

    layer: str
    reason: str
    score: float = 1.0


@dataclass(frozen=True)
class Verdict:
    action: Action
    source: Source
    name: str | None = None
    findings: tuple[Finding, ...] = field(default_factory=tuple)
    message: str | None = None   # safe to show the user; no scores or patterns

    @property
    def allowed(self) -> bool:
        return self.action is Action.ALLOW


# A detector inspects one item (and its normalized text) and returns findings.
# Detectors only report; policy decides what a finding means for each source.
Detector = Callable[[GuardInput, str], list[Finding]]
