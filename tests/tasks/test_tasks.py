import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.output_parsers import StrOutputParser

from src.tasks import get_task

STYLES = ["formal", "fiction"]
TASKS = ["grammar", "qa", "feedback"]
TEXT = "Their going to the market tomorow, and she dont know what too buy."
FAKE_REPLY = "Corrected text."


@pytest.fixture(autouse=True)
def no_chroma(monkeypatch):
    """Replace retrieval so no embedding model or Chroma collection is touched."""
    monkeypatch.setattr("src.tasks.base.retrieve", lambda *a, **k: [])
    monkeypatch.setattr("src.tasks.base.format_context", lambda results: "GUIDELINES")


@pytest.fixture
def make_task():
    """Build a real task, but swap its model for a fake that needs no server."""
    def _make(name: str, style: str = "formal"):
        task = get_task(name, style)
        task.model = FakeListChatModel(responses=[FAKE_REPLY])
        task.chain = task.model | StrOutputParser()
        return task
    return _make


def assert_reply(result):
    assert isinstance(result, str)
    assert result.strip()


@pytest.mark.parametrize("style", STYLES)
def test_grammar_returns_text(make_task, style):
    assert_reply(make_task("grammar", style).run(TEXT))


def test_grammar_returns_model_output(make_task):
    assert FAKE_REPLY in make_task("grammar").run(TEXT)


def test_grammar_adds_cut_note(make_task):
    task = make_task("grammar")
    task.max_words = 8   # instance override, the class stays unchanged
    long_text = "\n\n".join([TEXT] * 3)
    assert "input was cut" in task.run(long_text)


@pytest.mark.parametrize("style", STYLES)
def test_feedback_returns_text(make_task, style):
    assert_reply(make_task("feedback", style).run(TEXT))


@pytest.mark.parametrize("style", STYLES)
def test_qa_returns_text(make_task, style):
    assert_reply(make_task("qa", style).run("Who is Iku?"))


def test_qa_accepts_history(make_task):
    history = [
        {"role": "user", "task": "qa", "content": "Who is Iku?"},
        {"role": "assistant", "task": "qa", "content": "A character."},
    ]
    assert_reply(make_task("qa", "fiction").run("And Shin?", history=history))


@pytest.mark.parametrize("name", TASKS)
def test_build_prompt_contains_text(name):
    prompt = get_task(name, "formal").build_prompt(TEXT)
    assert TEXT in prompt