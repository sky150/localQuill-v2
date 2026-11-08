"""conftest.py runs before any tests and sets up the LM Studio fixture.
Other tests load it as a local plugin, ensuring ENV variables are set correctly.
Lastly, it makes fixtures available for the tests. """

import os
import httpx
import pytest

TEST_MODEL = "qwen2.5-coder-0.5b-instruct"
TASK_NAMES = ["TEST", "GRAMMAR", "QA", "FEEDBACK", "AUTOCOMPLETE"]


def force_test_environment():
    """Pin every task to the small model. Must run before src.config is imported."""
    for name in TASK_NAMES:
        os.environ[f"{name}_MODEL"] = TEST_MODEL
    os.environ["LIMIT_TOKEN_NUMBER"] = "50"   # short replies
    os.environ["VERBOSE"] = "true"            # full logs

force_test_environment()

@pytest.fixture(scope="session")
def ollama_ready():
    """Skip the Ollama tests if the server is down or the model is not pulled."""
    url = os.getenv("LMSTUDIO_URL", "http://127.0.0.1:1234/v1")
    try:
        response = httpx.get(f"{url}/api/tags", timeout=2)
        names = [m["name"] for m in response.json()["models"]]
    except httpx.HTTPError:
        pytest.skip("LM Studio is not running")
    if TEST_MODEL not in names:
        pytest.skip(f"Run: lmstudio pull {TEST_MODEL}")