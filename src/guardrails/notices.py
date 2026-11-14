"""Collect user-facing guardrail notices during one request.

Retrieval runs deep inside a task, in a worker thread, and cannot return notices up
the call stack without changing every signature. A context variable holds a list for
the current request instead. asyncio.to_thread copies the context, so the list set by
the route is the same list the worker thread appends to.

    with collect_notices() as notices:
        result = await asyncio.to_thread(task.run, ...)
    # notices now holds everything reported during the request
"""

from contextlib import contextmanager
from contextvars import ContextVar

from src.guardrails.types import Verdict

_current: ContextVar[list[dict] | None] = ContextVar("guardrail_notices", default=None)


def notice_from(verdict: Verdict) -> dict:
    return {
        "source": verdict.source.value,
        "name": verdict.name,
        "message": verdict.message or "",
    }


def report(verdict: Verdict) -> None:
    """Record a non-allowed verdict for the current request, once per message."""
    bucket = _current.get()
    if bucket is None or verdict.allowed:
        return
    entry = notice_from(verdict)
    if entry not in bucket:
        bucket.append(entry)


def report_text(source: str, message: str, name: str | None = None) -> None:
    """Record a notice that did not come from a scan (e.g. a withheld reply)."""
    bucket = _current.get()
    entry = {"source": source, "name": name, "message": message}
    if bucket is not None and entry not in bucket:
        bucket.append(entry)


@contextmanager
def collect_notices():
    bucket: list[dict] = []
    token = _current.set(bucket)
    try:
        yield bucket
    finally:
        _current.reset(token)
