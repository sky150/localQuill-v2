"""Integration tests for LM Studio.
LM Studio is required to be open in the background and the TEST_MODEL must be available.
"""
import shutil
import subprocess
import time
import warnings
from typing import NoReturn

import httpx
import pytest
from langchain_core.output_parsers import StrOutputParser

from src.llm.factory import get_model         
from src.config import LMSTUDIO_URL, get_config
from src.tasks import get_task

pytestmark = pytest.mark.integration

TEXT = "Their going to the market tomorow, and she dont know what too buy."


def _model_ids() -> list[str] | None:
    """Return the model ids LM Studio serves, or None if unreachable."""
    try:
        r = httpx.get(f"{LMSTUDIO_URL.rstrip('/')}/models", timeout=2)
        r.raise_for_status()
        return [m["id"] for m in r.json()["data"]]
    except (httpx.HTTPError, KeyError, ValueError):
        return None


def _try_start_server() -> bool:
    """Start the server through the lms CLI. False if the CLI is missing or fails."""
    if not shutil.which("lms"):
        return False
    try:
        subprocess.run(["lms", "server", "start"], timeout=30, check=True,
                       capture_output=True)
    except (subprocess.SubprocessError, OSError):
        return False
    for _ in range(15):                     # wait up to 15 s for the port
        if _model_ids() is not None:
            return True
        time.sleep(1)
    return False


def _skip_with_warning(msg: str) -> NoReturn:
    warnings.warn(msg, stacklevel=2)
    pytest.skip(msg)        


@pytest.fixture(scope="module")
def lmstudio_ready():
    started_by_test = False
    ids = _model_ids()
    if ids is None:
        started_by_test = _try_start_server()
        ids = _model_ids() if started_by_test else None
    if ids is None:
        _skip_with_warning(
            f"LM Studio not reachable at {LMSTUDIO_URL} and the server could not "
            "be started. Open LM Studio, or run `lms server start`."
        )

    model = get_config("test").model
    if model not in ids:
        _skip_with_warning(f"Model '{model}' is not available in LM Studio. Served: {ids}")
    

    yield model

    if started_by_test:                     # leave the server as we found it
        subprocess.run(["lms", "server", "stop"], capture_output=True)


@pytest.fixture
def no_chroma(monkeypatch):
    monkeypatch.setattr("src.tasks.base.retrieve", lambda *a, **k: [])
    monkeypatch.setattr("src.tasks.base.format_context", lambda r: "GUIDELINES")


def test_lmstudio_server_lists_models(lmstudio_ready):
    assert lmstudio_ready in _model_ids()


def test_grammar_task_runs_against_lmstudio(lmstudio_ready, no_chroma):
    task = get_task("test", "formal")
    task.model = get_model("test", provider="lmstudio")
    task.chain = task.model | StrOutputParser()

    result = task.run(TEXT)

    assert isinstance(result, str)
    assert result.strip()
    assert "Response time:" in result