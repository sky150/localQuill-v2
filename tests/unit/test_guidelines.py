import pytest

from src.config import STYLE_COLLECTIONS, get_config
from src.tasks.feedback import FeedbackTask
from src.tasks.grammar import GrammarTask


@pytest.mark.parametrize("task_cls", [GrammarTask, FeedbackTask])
@pytest.mark.parametrize("style", list(STYLE_COLLECTIONS))
def test_every_style_has_fixed_guidelines(task_cls, style, monkeypatch):
    """Grammar and feedback always include the hand-picked guidelines (retrieval adds to them)."""
    monkeypatch.setattr("src.tasks.feedback.retrieve_from", lambda *a, **k: "")
    task = task_cls(model=_echo(), config=get_config(task_cls.name, style), style=style)
    guidelines = task.fixed_guidelines()
    assert guidelines
    assert guidelines in task.build_prompt("Some draft text.")


def _echo():
    from langchain_core.runnables import RunnableLambda

    return RunnableLambda(lambda x: x)


@pytest.mark.parametrize("task", ["grammar", "feedback"])
def test_thinking_off_for_grammar_and_feedback(task):
    """qwen3.5 thinking alone filled num_ctx and left the answer empty (docs/rag-evaluation.md)."""
    from src.llm.factory import _build_ollama

    assert _build_ollama(get_config(task)).reasoning is False


def test_style_query_falls_back_to_keywords_when_the_model_fails():
    from langchain_core.runnables import RunnableLambda

    def fail(_):
        raise ConnectionError("model down")

    task = GrammarTask(model=RunnableLambda(fail), config=get_config("grammar"), style="fiction")
    assert task.style_query("Some draft text.") == GrammarTask.retrieval_query


def test_style_query_takes_the_first_label_from_the_list():
    from langchain_core.runnables import RunnableLambda

    reply = '- **No clear thesis or central claim**\n2. "Missing transitions between ideas."\n- made-up problem'
    task = FeedbackTask(model=RunnableLambda(lambda _: reply), config=get_config("feedback"), style="formal")
    assert task.style_query("Some draft text.") == "no clear thesis or central claim"


def test_style_query_falls_back_when_no_label_is_picked():
    from langchain_core.runnables import RunnableLambda

    task = FeedbackTask(model=RunnableLambda(lambda _: "The draft is fine."),
                        config=get_config("feedback"), style="fiction")
    assert task.style_query("Some draft text.") == FeedbackTask.retrieval_query


def test_references_are_marked_as_not_the_users_text():
    from langchain_core.documents import Document
    from src.rag.retriever import format_references

    doc = Document(page_content="Use transitions.", metadata={"source": "data/styles/formal/Guide.pdf"})
    out = format_references([(doc, 0.1), (doc, 0.2)])
    assert out.startswith('REFERENCE 1, from "Guide"') and "not the user's text" in out
    assert out.count("REFERENCE") == 1  # duplicates dropped


@pytest.mark.parametrize("task_cls", [GrammarTask, FeedbackTask])
def test_style_guide_search_off_when_collection_is_none(task_cls):
    """collection=None switches the style-guide search (and its extra model call) off."""
    from dataclasses import replace
    from langchain_core.runnables import RunnableLambda

    def no_call(_):
        raise AssertionError("the model must not be asked for a label")

    config = replace(get_config(task_cls.name), collection=None)
    task = task_cls(model=RunnableLambda(no_call), config=config, style="formal")
    assert task.search_guides("formal", "Some draft text.", top_k=2) == ""
