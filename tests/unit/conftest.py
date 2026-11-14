import pytest

from src.guardrails import reset_pipeline


@pytest.fixture(autouse=True)
def classifier_off_by_default(monkeypatch):
    """Unit tests must not load the 700 MB model. Patterns still run; tests that need the
    classifier pass in a fake scorer explicitly."""
    monkeypatch.setenv("GUARDRAIL_CLASSIFIER", "false")
    reset_pipeline()
    yield
    reset_pipeline()
