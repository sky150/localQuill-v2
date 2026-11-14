"""Every tunable value for the guardrails, in one place.

Nothing else in the package defines a threshold, limit or switch. Detectors, policy and
the call sites receive a `Settings` object, so tests build their own instead of
monkeypatching module constants, and per-style overrides have an obvious home.

Environment variables (all optional):

    USE_GUARDRAILS=false                  switch everything off
    GUARDRAIL_CLASSIFIER=false            regexes only, never load the model
    GUARDRAIL_MODEL=<hf repo>             classifier to use
    GUARDRAIL_DEVICE=cpu|mps              where the classifier runs
    GUARDRAIL_THRESHOLD_<SOURCE>=0.97     per source: USER_TEXT, ATTACHMENT, RETRIEVED,
                                          HISTORY, UPLOAD
"""

import os
from collections.abc import Mapping
from dataclasses import dataclass, field

from src.guardrails.types import Action, Source

DEFAULT_MODEL = "protectai/deberta-v3-base-prompt-injection-v2"


def _thresholds() -> dict[Source, float]:
    # The classifier is overconfident: scores cluster near 0 and near 1, so the catch rate
    # barely moves between 0.5 and 0.99 while false positives halve (12 -> 6 of 2,396
    # style-guide chunks, mostly PDF-extraction junk). Hence the high values. User text gets
    # a slightly lower bar: it is the direct attack surface and the user can just rephrase.
    # Provisional: re-measure with tests/guardrails/baseline.py after any change.
    return {
        Source.USER_TEXT: 0.95,
        Source.ATTACHMENT: 0.99,
        Source.RETRIEVED: 0.99,
        Source.HISTORY: 0.99,
        Source.UPLOAD: 0.99,
    }


def _actions() -> dict[Source, Action]:
    return {
        Source.USER_TEXT: Action.BLOCK,     # reject the request, ask to rephrase
        Source.ATTACHMENT: Action.DROP,     # leave the file out, say so
        Source.RETRIEVED: Action.DROP,      # leave the chunk out, say so
        Source.HISTORY: Action.DROP,        # forget the turn
        Source.UPLOAD: Action.DROP,         # indexed anyway; filtered again at retrieval
    }


@dataclass(frozen=True)
class Settings:
    # --- switches
    enabled: bool = True
    classifier_enabled: bool = True

    # --- classifier
    model: str = DEFAULT_MODEL
    device: str | None = None
    min_score: float = 0.5            # below this the classifier reports nothing at all
    window_chars: int = 1500          # ~350 tokens, safely under the model's 512-token limit
    window_overlap: int = 200         # a sentence cut at a boundary still appears whole once
    max_windows: int = 60             # upper bound on work per item (~90k characters)
    score_cache_size: int = 2048

    # --- patterns
    max_single_line_chars: int = 20000   # user text only: a huge one-line payload is a smell

    # --- policy: what a finding means for each source
    thresholds: Mapping[Source, float] = field(default_factory=_thresholds)
    actions: Mapping[Source, Action] = field(default_factory=_actions)

    # --- limits at the call sites
    max_history_scanned: int = 6         # prompts only read the latest few turns
    max_upload_scan_chunks: int = 500    # ~0.06 s each; retrieval scans whatever this skips
    untrusted_collections: frozenset[str] = frozenset({"userdocs", "outline"})

    def threshold(self, source: Source) -> float:
        return self.thresholds[source]

    def action(self, source: Source) -> Action:
        return self.actions[source]

    @classmethod
    def from_env(cls) -> "Settings":
        thresholds = _thresholds()
        for source in Source:
            value = os.getenv(f"GUARDRAIL_THRESHOLD_{source.name}", "").strip()
            if value:
                thresholds[source] = float(value)
        return cls(
            enabled=_env_bool("USE_GUARDRAILS", True),
            classifier_enabled=_env_bool("GUARDRAIL_CLASSIFIER", True),
            model=os.getenv("GUARDRAIL_MODEL", "").strip() or DEFAULT_MODEL,
            device=os.getenv("GUARDRAIL_DEVICE", "").strip() or None,
            thresholds=thresholds,
        )


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return value.strip().lower() != "false"
