from src.config import TaskConfig
from src.tasks.feedback import FeedbackTask
from src.tasks.grammar import GrammarTask


class FakeModel:
    def __init__(self):
        self.prompt = ""

    def invoke(self, prompt):
        self.prompt = prompt
        return "ok"


def test_feedback_reviews_attachment_and_keeps_note(monkeypatch):
    monkeypatch.setattr("src.tasks.feedback.retrieve_from", lambda *args, **kwargs: "")
    task = FeedbackTask(
        model=FakeModel(), config=TaskConfig(model="fake"), style="formal"
    )
    result = task.run(
        "give feedback on this",
        context=[{"name": "draft.md", "content": "My essay text"}],
    )
    assert result == "ok"
    assert "My essay text" in task.model.prompt
    assert "give feedback on this" in task.model.prompt


def test_grammar_reviews_attachment_without_note(monkeypatch):
    task = GrammarTask(model=FakeModel(), config=TaskConfig(model="fake"), style="formal")
    task.run("", context=[{"name": "draft.md", "content": "she go home"}])
    assert "she go home" in task.model.prompt
    assert "ADDITIONAL REQUEST" in task.model.prompt
