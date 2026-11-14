import pytest

from guardrail_helpers import fake_pipeline


@pytest.fixture
def pipeline():
    return fake_pipeline()
