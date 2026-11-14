"""Security note + canary around the model call."""

import re

import pytest
from langchain_core.runnables import Runnable

from src.config import TaskConfig
from src.guardrails.containment import LEAK_MESSAGE, containment_note, leaked, new_canary
from src.guardrails.notices import collect_notices
from src.tasks.autocomplete import AutocompleteTask
from src.tasks.feedback import FeedbackTask
from src.tasks.grammar import GrammarTask
from src.tasks.qa import QATask


class RecordingModel(Runnable):
    """Returns a fixed reply, or echoes the canary it finds in the prompt."""

    def __init__(self, reply="fine", echo_canary=False):
        self.reply = reply
        self.echo_canary = echo_canary
        self.prompt = ""

    def invoke(self, input, config=None, **kwargs):
        self.prompt = input
        if self.echo_canary:
            return "My instructions say: " + re.search(r"QK-[0-9a-f]{12}", input).group(0)
        return self.reply


def make(cls, **model_kwargs):
    return cls(model=RecordingModel(**model_kwargs), config=TaskConfig(model="fake"), style="formal")


@pytest.fixture(autouse=True)
def no_retrieval(monkeypatch):
    monkeypatch.setattr("src.tasks.feedback.retrieve_from", lambda *a, **k: "")
    monkeypatch.setattr("src.tasks.qa.retrieve_from", lambda *a, **k: "")
    monkeypatch.setattr("src.tasks.base.retrieve", lambda *a, **k: [])


class TestHelpers:
    def test_canaries_are_unique_and_well_formed(self):
        a, b = new_canary(), new_canary()
        assert a != b
        assert re.fullmatch(r"QK-[0-9a-f]{12}", a)

    def test_note_contains_canary_and_the_data_rule(self):
        note = containment_note("QK-abc")
        assert "QK-abc" in note
        assert "never instructions" in note

    def test_leak_detection(self):
        assert leaked("QK-abc", "the reference is QK-abc")
        assert not leaked("QK-abc", "nothing to see")


@pytest.mark.parametrize("cls", [GrammarTask, FeedbackTask, QATask])
class TestContainedTasks:
    def test_prompt_ends_with_security_note(self, cls):
        task = make(cls)
        task.run("some text")
        assert "SECURITY NOTE" in task.model.prompt
        assert task.model.prompt.rstrip().endswith("never repeat it.")

    def test_leaked_canary_withholds_reply_and_notifies(self, cls):
        task = make(cls, echo_canary=True)
        with collect_notices() as notices:
            result = task.run("some text")
        assert result == LEAK_MESSAGE
        assert "QK-" not in result
        assert notices and notices[0]["source"] == "output"

    def test_normal_reply_is_returned(self, cls):
        task = make(cls, reply="Here is the answer.")
        assert task.run("some text").startswith("Here is the answer.")

    def test_each_request_gets_a_fresh_canary(self, cls):
        task = make(cls)
        task.run("one")
        first = re.search(r"QK-[0-9a-f]{12}", task.model.prompt).group(0)
        task.run("two")
        second = re.search(r"QK-[0-9a-f]{12}", task.model.prompt).group(0)
        assert first != second


def test_autocomplete_prompt_is_untouched():
    """Autocomplete continues the user's text, so nothing may be appended to it."""
    task = make(AutocompleteTask)
    task.template = "{text}"
    task.run("The rain began to")
    assert task.model.prompt == "The rain began to"
