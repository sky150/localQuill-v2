import pytest
from src.tasks import get_task


pytestmark = pytest.mark.usefixtures("ollama_ready")

STYLES = ["formal", "fiction"]
TASKS = ["grammar", "qa", "feedback"]
TEXT = "Their going to the market tomorow, and she dont know what too buy."


def assert_reply(result):
    assert isinstance(result, str)
    assert result.strip()


@pytest.mark.parametrize("name", TASKS)
def test_tasks_use_smallest_model(name):
    assert get_task(name, "formal").model.model == "qwen2.5:0.5b"


@pytest.mark.parametrize("style", STYLES)
def test_grammar_returns_text(style):
    assert_reply(get_task("grammar", style).run(TEXT))


def test_grammar_adds_cut_note():
    task = get_task("grammar", "formal")
    task.max_words = 8   # instance override, the class stays unchanged
    long_text = "\n\n".join([TEXT] * 3)
    assert "input was cut" in task.run(long_text)


@pytest.mark.parametrize("style", STYLES)
def test_feedback_returns_text(style):
    assert_reply(get_task("feedback", style).run(TEXT))


@pytest.mark.parametrize("style", STYLES)
def test_qa_returns_text(style):
    assert_reply(get_task("qa", style).run("Who is Iku?"))


def test_qa_accepts_history():
    history = [
        {"role": "user", "task": "qa", "content": "Who is Iku?"},
        {"role": "assistant", "task": "qa", "content": "A character."},
    ]
    assert_reply(get_task("qa", "fiction").run("And Shin?", history=history))


@pytest.mark.parametrize("name", TASKS)
def test_build_prompt_contains_text(name):
    prompt = get_task(name, "formal").build_prompt(TEXT)
    assert TEXT in prompt