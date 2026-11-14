"""Measure the guardrails: injection catch rate and false positives on real writing.

    .venv/bin/python tests/guardrails/baseline.py                # regexes only (fast)
    .venv/bin/python tests/guardrails/baseline.py --classifier   # full pipeline (~3 min)

Benign text = 1,000-character chunks of the .md and .pdf files under data/styles.
Direct injections are scanned as user text, indirect ones as attachments.
Re-run after every guardrail change and compare the numbers.
"""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.guardrails import GuardInput, GuardrailPipeline, Settings, Source  # noqa: E402
from src.guardrails.detectors import ClassifierLayer, PatternLayer  # noqa: E402

CHUNK_CHARS = 1000


def chunks(text: str, size: int = CHUNK_CHARS):
    for start in range(0, len(text), size):
        yield text[start : start + size]


def read_text(path: Path) -> str:
    if path.suffix == ".pdf":
        from pypdf import PdfReader

        return "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
    return path.read_text(errors="ignore")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--classifier", action="store_true", help="include the model")
    parser.add_argument("--quiet", action="store_true", help="hide per-file lines")
    args = parser.parse_args()

    settings = Settings.from_env()   # picks up GUARDRAIL_THRESHOLD_* so you can try values
    layers = [PatternLayer(settings)]
    if args.classifier:
        layers.append(ClassifierLayer(settings))
    pipeline = GuardrailPipeline(layers=layers, settings=settings)
    print("layers:", [type(layer).__name__ for layer in layers])
    print("thresholds:", {s.value: t for s, t in settings.thresholds.items()})

    samples = json.loads((Path(__file__).parent / "injection_samples.json").read_text())
    print("\n== Injection samples (want: flagged) ==")
    for kind in ("direct", "indirect"):
        source = Source.USER_TEXT if kind == "direct" else Source.ATTACHMENT
        texts = samples[kind]
        missed = [t for t in texts if pipeline.scan(GuardInput(source, t)).allowed]
        caught = len(texts) - len(missed)
        print(f"{kind:9s} caught {caught}/{len(texts)} ({caught / len(texts):.0%})")
        for t in missed:
            print(f"   missed: {t[:90]}")

    print("\n== Benign writing (want: allowed) ==")
    total = flagged = 0
    examples: list[tuple[str, str, str]] = []
    start = time.time()
    styles = ROOT / "data" / "styles"
    for path in sorted([*styles.rglob("*.md"), *styles.rglob("*.pdf")]):
        file_total = file_flagged = 0
        for chunk in chunks(read_text(path)):
            verdict = pipeline.scan(GuardInput(Source.ATTACHMENT, chunk, name=path.name))
            file_total += 1
            if not verdict.allowed:
                file_flagged += 1
                if len(examples) < 10:
                    examples.append((path.name, verdict.findings[0].reason, chunk.strip()[:110]))
        total += file_total
        flagged += file_flagged
        if not args.quiet:
            print(f"{path.name[:60]:60s} {file_flagged:3d}/{file_total:<4d} flagged")
    rate = flagged / total if total else 0
    print(f"\nFalse positives: {flagged}/{total} chunks ({rate:.2%}) in {time.time() - start:.0f}s")
    for name, reason, snippet in examples:
        print(f"  {name[:30]} | {reason}\n    {snippet!r}")
    print(
        "\nNote: this benign set has little dialogue-heavy fiction, so the rate is a floor "
        "for fiction writers."
    )


if __name__ == "__main__":
    main()
