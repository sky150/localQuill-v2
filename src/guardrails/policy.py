from src.guardrails.config import Settings
from src.guardrails.messages import message_for
from src.guardrails.types import Action, Finding, GuardInput, Verdict


def decide(item: GuardInput, findings: list[Finding], settings: Settings) -> Verdict:
    """Pure function: findings + settings -> verdict.

    A finding triggers the source's action once its score reaches the source's threshold.
    Pattern findings score 1.0 and always trigger. Lower scores stay on the verdict
    (for logging and tuning) but the item is allowed.
    """
    threshold = settings.threshold(item.source)
    triggering = [f for f in findings if f.score >= threshold]
    if not triggering:
        return Verdict(Action.ALLOW, item.source, item.name, tuple(findings))
    return Verdict(
        settings.action(item.source),
        item.source,
        item.name,
        tuple(triggering),
        message_for(item.source, item.name),
    )
