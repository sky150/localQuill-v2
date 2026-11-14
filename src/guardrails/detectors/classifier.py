"""Layer 2: a small local prompt-injection classifier, scored over overlapping windows.

The model is loaded lazily, from the local Hugging Face cache only, so the app never
reaches for the network at request time. Fetch it once at setup:

    python -m src.guardrails.download

If the model is not available the layer switches itself off with one warning and the
pipeline carries on with the other layers. Chat keeps working, with weaker protection.
"""

import hashlib
import logging
from collections import OrderedDict
from collections.abc import Callable, Sequence

from src.guardrails.config import Settings
from src.guardrails.types import Finding, GuardInput

logger = logging.getLogger(__name__)

_DEFAULTS = Settings()

# A scorer maps texts to injection probabilities in [0, 1]. Tests pass in a fake one.
Scorer = Callable[[Sequence[str]], list[float]]


def make_windows(
    text: str,
    size: int = _DEFAULTS.window_chars,
    overlap: int = _DEFAULTS.window_overlap,
    max_windows: int = _DEFAULTS.max_windows,
) -> list[str]:
    """Split text into overlapping windows, preferring to cut at whitespace."""
    if len(text) <= size:
        return [text]
    windows, start = [], 0
    while start < len(text) and len(windows) < max_windows:
        end = min(start + size, len(text))
        if end < len(text):
            cut = text.rfind(" ", start + size - overlap, end)
            end = cut if cut > start else end
        windows.append(text[start:end])
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return windows


class HFInjectionScorer:
    """Wraps the Hugging Face text-classification pipeline. Loads on first use."""

    def __init__(self, model: str, device: str | None = None):
        self.model = model
        self.device = device
        self._pipe = None

    def _load(self):
        from huggingface_hub import snapshot_download
        from transformers import pipeline

        # Resolve the cached snapshot; raises if it was never downloaded.
        # This keeps model and tokenizer off the network at request time.
        path = snapshot_download(self.model, local_files_only=True)
        kwargs = {"model": path, "truncation": True, "max_length": 512}
        if self.device:
            kwargs["device"] = self.device
        self._pipe = pipeline("text-classification", **kwargs)
        return self._pipe

    @property
    def pipe(self):
        return self._pipe or self._load()

    def __call__(self, texts: Sequence[str]) -> list[float]:
        results = self.pipe(list(texts), batch_size=8)
        scores = []
        for r in results:
            positive = r["label"].upper() == "INJECTION"
            scores.append(r["score"] if positive else 1.0 - r["score"])
        return scores


class ClassifierLayer:
    """Scores every window and reports the worst one."""

    name = "classifier"

    def __init__(
        self,
        settings: Settings | None = None,
        scorer: Scorer | None = None,
        min_score: float | None = None,
        sources=None,
    ):
        self.settings = settings or Settings()
        self._scorer = scorer
        self.min_score = self.settings.min_score if min_score is None else min_score
        self.sources = sources         # None = every source
        self.available = True
        self._cache: OrderedDict[str, float] = OrderedDict()

    @property
    def scorer(self) -> Scorer:
        if self._scorer is None:
            self._scorer = HFInjectionScorer(self.settings.model, self.settings.device)
        return self._scorer

    def _score(self, windows: list[str]) -> list[float]:
        keys = [hashlib.sha1(w.encode()).hexdigest() for w in windows]
        missing = [(k, w) for k, w in zip(keys, windows) if k not in self._cache]
        if missing:
            for (key, _), score in zip(missing, self.scorer([w for _, w in missing])):
                self._cache[key] = score
                if len(self._cache) > self.settings.score_cache_size:
                    self._cache.popitem(last=False)
        return [self._cache[k] for k in keys]

    def __call__(self, item: GuardInput, text: str) -> list[Finding]:
        if not self.available or (self.sources and item.source not in self.sources):
            return []
        try:
            s = self.settings
            scores = self._score(
                make_windows(text, s.window_chars, s.window_overlap, s.max_windows)
            )
        except Exception as error:  # model missing, out of memory, bad install...
            self.available = False
            logger.warning(
                "[guardrails] classifier disabled (%s: %s). Run "
                "'python -m src.guardrails.download' to fetch the model.",
                type(error).__name__,
                error,
            )
            return []
        worst = max(scores)
        if worst < self.min_score:
            return []
        return [Finding(self.name, f"injection score {worst:.2f}", worst)]
