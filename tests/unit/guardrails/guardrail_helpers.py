"""Fakes shared by the guardrail unit tests. No model is ever loaded."""

from src.guardrails import GuardrailPipeline, Settings
from src.guardrails.detectors import ClassifierLayer, PatternLayer


class FakeScorer:
    """Scores 0.99 for any window containing PAYLOAD, else 0.01. Records its calls."""

    def __init__(self, hit="PAYLOAD"):
        self.hit = hit
        self.calls: list[list[str]] = []

    def __call__(self, texts):
        self.calls.append(list(texts))
        return [0.99 if self.hit in t else 0.01 for t in texts]


def fake_pipeline(settings: Settings | None = None) -> GuardrailPipeline:
    """Regexes on user text, plus a classifier that flags anything containing PAYLOAD."""
    settings = settings or Settings()
    return GuardrailPipeline(
        layers=[PatternLayer(settings), ClassifierLayer(settings, scorer=FakeScorer())],
        settings=settings,
    )
