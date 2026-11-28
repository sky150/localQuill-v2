"""Offline tests for the composite ranking of RAG vs. no-RAG results (ranking_rag_norag).

The shipped qna and feedback weights are 0.0 placeholders, so tests set their own weights and
gates through monkeypatch.setitem.
"""
import math

import pandas as pd
import pytest

from src.evaluation import ranking_rag_norag as rk
from src.evaluation.common import UnsupportedPrompt


def _grouped(rows):
    """rows: (model, setting_id, {metric: mean}) -> long frame like load_grouped returns."""
    recs = []
    for model, setting_id, metrics in rows:
        temp = pd.to_numeric(setting_id.removeprefix("t"), errors="coerce")
        for metric, mean in metrics.items():
            recs.append({"metric": metric, "setting_id": setting_id, "temperature": temp,
                         "model": model, "mean": mean})
    return pd.DataFrame(recs)


def _row(ranked, model, setting_id):
    return ranked[(ranked["model"] == model) & (ranked["setting_id"] == setting_id)].iloc[0]


@pytest.fixture
def qna_weights(monkeypatch):
    monkeypatch.setitem(rk.WEIGHTS_BY_PROMPT, "qna", {
        "judge_correctness": 0.4, "judge_groundedness": 0.4, "gen_tok_per_s": 0.2})
    monkeypatch.setitem(rk.GATES_BY_PROMPT, "qna", {"capped_rate": 0.1})


@pytest.fixture
def feedback_weights(monkeypatch):
    monkeypatch.setitem(rk.WEIGHTS_BY_PROMPT, "feedback", {
        "judge_clarity": 0.5, "judge_style": 0.3, "gen_tok_per_s": 0.2})
    monkeypatch.setitem(rk.GATES_BY_PROMPT, "feedback", {})


def test_qna_better_judge_scores_rank_first(qna_weights):
    g = _grouped([
        ("good", "t0.2", {"judge_correctness": 4.5, "judge_groundedness": 4.6,
                          "gen_tok_per_s": 50.0, "capped_rate": 0.0}),
        ("bad", "t0.2", {"judge_correctness": 3.0, "judge_groundedness": 3.2,
                         "gen_tok_per_s": 50.0, "capped_rate": 0.0}),
    ])
    ranked = rk.composite(g, prompt="qna")
    assert ranked.iloc[0]["model"] == "good"
    assert _row(ranked, "good", "t0.2")["rank"] == 1
    assert _row(ranked, "bad", "t0.2")["rank"] == 2


def test_qna_row_without_tok_per_s_still_scored(qna_weights):
    g = _grouped([
        ("local", "t0.2", {"judge_correctness": 4.0, "judge_groundedness": 4.0,
                           "gen_tok_per_s": 40.0, "capped_rate": 0.0}),
        ("slow", "t0.2", {"judge_correctness": 3.0, "judge_groundedness": 3.0,
                          "gen_tok_per_s": 20.0, "capped_rate": 0.0}),
        ("gpt-5-nano", "effort_low", {"judge_correctness": 4.5, "judge_groundedness": 3.5,
                                      "capped_rate": 0.0}),
    ])
    ranked = rk.composite(g, prompt="qna")
    score = _row(ranked, "gpt-5-nano", "effort_low")["score"]
    assert not math.isnan(score)
    assert 0.0 <= score <= 1.0


@pytest.mark.parametrize("capped, passes", [
    (0.11, False),
    (0.1, True),
    (0.0, True),
])
def test_qna_capped_rate_gate(qna_weights, capped, passes):
    g = _grouped([
        ("a", "t0.2", {"judge_correctness": 4.0, "judge_groundedness": 4.0,
                       "gen_tok_per_s": 40.0, "capped_rate": capped}),
        ("b", "t0.2", {"judge_correctness": 3.0, "judge_groundedness": 3.0,
                       "gen_tok_per_s": 20.0, "capped_rate": 0.0}),
    ])
    row = _row(rk.composite(g, prompt="qna"), "a", "t0.2")
    assert bool(row["passed_gate"]) is passes
    if passes:
        assert not math.isnan(row["score"])
        assert pd.notna(row["rank"])
    else:
        assert math.isnan(row["score"])
        assert pd.isna(row["rank"])


@pytest.mark.parametrize("cos_doc, passes", [(0.94, False), (0.95, True), (0.99, True)])
def test_grammar_cos_doc_gate(cos_doc, passes):
    weights = rk.WEIGHTS_BY_PROMPT["grammar"]
    metric = next(iter(weights))
    g = _grouped([
        ("a", "t0.2", {"cos_doc": cos_doc, metric: 0.5}),
        ("b", "t0.2", {"cos_doc": 0.99, metric: 0.4}),
    ])
    row = _row(rk.composite(g), "a", "t0.2")
    assert bool(row["passed_gate"]) is passes
    assert math.isnan(row["score"]) is (not passes)
    assert pd.isna(row["rank"]) is (not passes)


def test_one_row_condition_scores_one_and_warns(feedback_weights, capsys):
    g = _grouped([
        ("only", "t0.2", {"judge_clarity": 0.8, "judge_style": 0.7, "gen_tok_per_s": 30.0}),
    ])
    ranked = rk.composite(g, prompt="feedback")
    assert len(ranked) == 1
    assert ranked.iloc[0]["score"] == pytest.approx(1.0)
    assert ranked.iloc[0]["rank"] == 1
    assert "pass the gate" in capsys.readouterr().out


def test_two_rows_pass_no_warning(feedback_weights, capsys):
    g = _grouped([
        ("a", "t0.2", {"judge_clarity": 0.8, "judge_style": 0.7, "gen_tok_per_s": 30.0}),
        ("b", "t0.2", {"judge_clarity": 0.6, "judge_style": 0.5, "gen_tok_per_s": 20.0}),
    ])
    rk.composite(g, prompt="feedback")
    assert "pass the gate" not in capsys.readouterr().out


def test_main_unknown_prompt_raises(monkeypatch):
    monkeypatch.delitem(rk.WEIGHTS_BY_PROMPT, "feedback")
    monkeypatch.setattr("sys.argv", ["ranking", "--style", "formal", "--prompt", "feedback"])
    with pytest.raises(UnsupportedPrompt, match="WEIGHTS_BY_PROMPT"):
        rk.main()


def test_metric_ranking_sorts_capped_rate_ascending():
    g = _grouped([
        ("high", "t0.2", {"capped_rate": 0.3}),
        ("low", "t0.2", {"capped_rate": 0.05}),
    ])
    ranked = rk.metric_ranking(g, "capped_rate")
    assert list(ranked["model"]) == ["low", "high"]
    assert rk.direction("capped_rate") == "lower = better"
