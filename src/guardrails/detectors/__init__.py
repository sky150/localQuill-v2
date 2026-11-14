"""Detectors find suspicious text and report findings. They never decide what happens next;
that is policy's job. Cheapest first: patterns, then the classifier."""

from src.guardrails.config import Settings
from src.guardrails.detectors.classifier import ClassifierLayer
from src.guardrails.detectors.patterns import PatternLayer
from src.guardrails.types import Detector


def build_detectors(settings: Settings) -> list[Detector]:
    detectors: list[Detector] = [PatternLayer(settings)]
    if settings.classifier_enabled:
        detectors.append(ClassifierLayer(settings))
    return detectors


__all__ = ["ClassifierLayer", "PatternLayer", "build_detectors"]
