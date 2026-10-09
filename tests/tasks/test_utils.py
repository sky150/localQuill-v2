# import utility import truncate_by_paragraph

from src.tasks.utility import truncate_by_paragraph
from src.tasks.qa import QATask


def test_truncate_keeps_whole_paragraphs():
    text = "one two three\n\nfour five six\n\nseven eight nine"
    result, was_cut = truncate_by_paragraph(text, max_words=6)
    assert was_cut
    assert result == "one two three\n\nfour five six"


def test_truncate_no_cut_when_short():
    result, was_cut = truncate_by_paragraph("a b c", max_words=10)
    assert not was_cut and result == "a b c"


def test_truncate_cuts_oversized_first_paragraph():
    result, was_cut = truncate_by_paragraph("a b c d e f", max_words=3)
    assert was_cut and result == "a b c"


def test_qa_history_ignores_other_tasks():
    history = [{"role": "assistant", "task": "grammar", "content": "corrected text"}]
    assert QATask._format_history(history) == "none"


def test_qa_history_keeps_qa_turns():
    history = [{"role": "user", "task": "qa", "content": "Who is Iku?"}]
    assert "Who is Iku?" in QATask._format_history(history)