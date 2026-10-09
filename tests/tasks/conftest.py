"""conftest.py runs before any tests and sets up the Ollama fixture.
Other tests load it as a local plugin, ensuring ENV variables are set correctly.
Lastly, it makes fixtures available for the tests. """

import os
import httpx
import pytest

TEST_MODEL = "qwen2.5:0.5b"
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
    url = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
    try:
        response = httpx.get(f"{url}/api/tags", timeout=2)
        names = [m["name"] for m in response.json()["models"]]
    except httpx.HTTPError:
        pytest.skip("Ollama is not running")
    if TEST_MODEL not in names:
        pytest.skip(f"Run: ollama pull {TEST_MODEL}")