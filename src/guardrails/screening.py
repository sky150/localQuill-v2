"""The only module the rest of the app needs to call.

Each function answers one question a call site has, and handles scanning, dropping and
telling the user. Callers never touch `Source`, thresholds or notices.

All functions take an optional `pipeline` so tests can pass their own.
"""

import os
from collections.abc import Sequence

from src.guardrails.notices import notice_from, report
from src.guardrails.pipeline import GuardrailPipeline, get_pipeline
from src.guardrails.types import GuardInput, Source, Verdict


def _use(pipeline: GuardrailPipeline | None) -> GuardrailPipeline:
    return pipeline or get_pipeline()


def screen_user_text(text: str, pipeline: GuardrailPipeline | None = None) -> Verdict:
    """Check what the user typed. Not allowed -> the caller rejects the request and shows
    `verdict.message`."""
    return _use(pipeline).scan(GuardInput(Source.USER_TEXT, text))


def screen_attachments(
    items: Sequence[dict], pipeline: GuardrailPipeline | None = None
) -> list[dict]:
    """Keep attachments that pass. Each item is {"name", "content", ...}; call this with
    the content that would actually reach the model (already clipped)."""
    pipe, kept = _use(pipeline), []
    for item in items:
        verdict = pipe.scan(
            GuardInput(Source.ATTACHMENT, item.get("content") or "", item.get("name"))
        )
        if verdict.allowed:
            kept.append(item)
        else:
            report(verdict)
    return kept


def screen_history(history: Sequence[dict], pipeline: GuardrailPipeline | None = None) -> list[dict]:
    """The client sends history back, so a forged assistant turn is possible. Only the most
    recent turns are scanned (and returned): no task reads further back."""
    pipe = _use(pipeline)
    start = max(len(history) - pipe.settings.max_history_scanned, 0)
    kept = []
    for message in history[start:]:
        verdict = pipe.scan(GuardInput(Source.HISTORY, message.get("content") or ""))
        if verdict.allowed:
            kept.append(message)
        else:
            report(verdict)
    return kept


def screen_chunks(results, collection: str, pipeline: GuardrailPipeline | None = None):
    """Drop retrieved (doc, score) pairs that look like instructions. Collections we curate
    ourselves (the style guides) are trusted and returned untouched."""
    pipe = _use(pipeline)
    if collection not in pipe.settings.untrusted_collections:
        return results
    kept = []
    for doc, score in results:
        name = os.path.basename(doc.metadata.get("source", "")) or collection
        verdict = pipe.scan(GuardInput(Source.RETRIEVED, doc.page_content, name))
        if verdict.allowed:
            kept.append((doc, score))
        else:
            report(verdict)
    return kept


def screen_upload_chunks(
    chunks, name: str, pipeline: GuardrailPipeline | None = None
) -> list[dict]:
    """Early warning at upload. Flagged files are still indexed (one false positive must not
    reject a whole novel) and are filtered again at retrieval. Returns notices, at most one
    per file. Scans only the first `max_upload_scan_chunks`; retrieval covers the rest."""
    pipe = _use(pipeline)
    for chunk in chunks[: pipe.settings.max_upload_scan_chunks]:
        verdict = pipe.scan(GuardInput(Source.UPLOAD, chunk.page_content, name))
        if not verdict.allowed:
            return [notice_from(verdict)]
    return []
