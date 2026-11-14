import dataclasses
import logging
from collections.abc import Iterable

from src.guardrails.config import Settings
from src.guardrails.detectors import build_detectors
from src.guardrails.normalize import normalize
from src.guardrails.policy import decide
from src.guardrails.types import Action, Detector, Finding, GuardInput, Verdict

logger = logging.getLogger(__name__)


class GuardrailPipeline:
    """normalize -> detectors (cheapest first) -> policy.

    Stops at the first detector whose finding reaches the source's threshold, so the
    classifier only runs on text the regexes let through. Findings below the threshold are
    kept on the verdict so tuning can see them.
    """

    def __init__(
        self,
        layers: Iterable[Detector] | None = None,
        settings: Settings | None = None,
        enabled: bool | None = None,
    ):
        self.settings = settings or Settings.from_env()
        self.layers = list(layers) if layers is not None else build_detectors(self.settings)
        self.enabled = self.settings.enabled if enabled is None else enabled

    def scan(self, item: GuardInput) -> Verdict:
        if not self.enabled or not item.text or not item.text.strip():
            return Verdict(Action.ALLOW, item.source, item.name)

        text = normalize(item.text)
        threshold = self.settings.threshold(item.source)
        findings: list[Finding] = []
        for layer in self.layers:
            found = layer(item, text)
            findings.extend(found)
            if any(f.score >= threshold for f in found):
                break

        verdict = decide(item, findings, self.settings)
        if not verdict.allowed:
            logger.warning(
                "[guardrails] %s source=%s name=%s findings=%s",
                verdict.action.value,
                item.source.value,
                item.name,
                [f"{f.layer}: {f.reason}" for f in verdict.findings],
            )
        return verdict

    def scan_many(self, items: Iterable[GuardInput]) -> list[Verdict]:
        return [self.scan(item) for item in items]


_pipeline: GuardrailPipeline | None = None


def get_pipeline() -> GuardrailPipeline:
    """The shared pipeline. Settings are re-read from the environment on every call, and
    the pipeline is rebuilt only when something other than the on/off switch changed, so
    the loaded model survives toggling USE_GUARDRAILS."""
    global _pipeline
    settings = Settings.from_env()
    if _pipeline is not None:
        same = dataclasses.replace(_pipeline.settings, enabled=True) == dataclasses.replace(
            settings, enabled=True
        )
        if same:
            _pipeline.settings = settings
            _pipeline.enabled = settings.enabled
            return _pipeline
    _pipeline = GuardrailPipeline(settings=settings)
    return _pipeline


def reset_pipeline() -> None:
    """Forget the shared pipeline (used by tests)."""
    global _pipeline
    _pipeline = None
