import re
from collections.abc import Iterable

from src.guardrails.config import Settings
from src.guardrails.types import Finding, GuardInput, Source

# Originally from ContentFilterMiddleware. Noisy on documents, which is why they only apply
# to user text by default; the classifier judges everything else.
BLOCKLIST = [
    r"ignore.*?previous.*(instructions|context|prompts)",
    r"disregard.*(previous|above).*(instructions|context|prompts)",
    r"forget.*?\b(previous|above)\b",
    r"override.*system instructions",
    # Only "follow these/my/new instructions". A bare "follow the instructions" is ordinary
    # writing advice ("follow the instructions in the style guide").
    r"follow\s+(only\s+)?(these|my|the\s+following|the\s+new|new)\s+instructions",
    r"you are now.*?\bassistant\b",
    r"disobey",
    r"bypass.*security|bypass the",
    r"insert code to",
    r"execute the following",
    r"run this code",
]


def compile_patterns(patterns: Iterable[str]) -> list[re.Pattern]:
    return [re.compile(p, re.IGNORECASE) for p in patterns]


_DEFAULT = compile_patterns(BLOCKLIST)


def find_matches(text: str, compiled: list[re.Pattern] | None = None) -> list[str]:
    """Return the source of every pattern that matches."""
    return [c.pattern for c in (compiled or _DEFAULT) if c.search(text)]


class PatternLayer:
    """Layer 1: cheap regex checks. Layers are callables: GuardInput -> list[Finding]."""

    name = "patterns"

    def __init__(
        self,
        settings: Settings | None = None,
        extra_blocklist: Iterable[str] = (),
        sources: Iterable[Source] | None = (Source.USER_TEXT,),
    ):
        """sources limits where the regexes apply; None means every source.

        The default is user text only. In documents the patterns match ordinary prose
        ("follow the instructions", "disobey"), and the classifier judges those better.
        """
        self.settings = settings or Settings()
        self._compiled = compile_patterns([*BLOCKLIST, *extra_blocklist])
        self.sources = None if sources is None else frozenset(sources)

    def __call__(self, item: GuardInput, text: str) -> list[Finding]:
        if self.sources is not None and item.source not in self.sources:
            return []
        findings = [
            Finding(self.name, f"matched {pattern}")
            for pattern in find_matches(text, self._compiled)
        ]
        # A huge single-line payload is a user-text smell. Real documents can be long.
        if (
            item.source is Source.USER_TEXT
            and len(text) > self.settings.max_single_line_chars
            and text.count("\n") < 2
        ):
            findings.append(Finding(self.name, "long single-line input"))
        return findings
