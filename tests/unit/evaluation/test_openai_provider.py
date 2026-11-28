"""Offline tests for the OpenAI (gpt-5-nano) provider of the evaluation harness.

No network and no API key: the payload check only builds the request dict.
"""
import math

import pandas as pd
import pytest
from langchain_openai import ChatOpenAI

from src.evaluation import config, ranking_rag_norag
from src.evaluation.generation import _invoke_kwargs

EFFORTS = [{"id": "effort_low", "reasoning_effort": "low"},
           {"id": "effort_medium", "reasoning_effort": "medium"}]


@pytest.mark.parametrize("prompt", ["grammar", "qna"])
def test_settings_for_openai_returns_efforts(prompt):
    assert config.settings_for(prompt, "openai") == EFFORTS


@pytest.mark.parametrize("prompt", sorted(config.SETTINGS_BY_PROMPT))
def test_settings_for_ollama_unchanged(prompt):
    assert config.settings_for(prompt, "ollama") == config.SETTINGS_BY_PROMPT[prompt]


@pytest.mark.parametrize("setting, expected", [
    ({"id": "effort_low", "reasoning_effort": "low"}, config.N_RUNS),
    ({"id": "t0.0", "temperature": 0}, 1),
    ({"id": "t0.2", "temperature": 0.2}, config.N_RUNS),
])
def test_runs_for(setting, expected):
    assert config.runs_for(setting, config.N_RUNS) == expected


def test_models_for_feedback_openai_raises():
    with pytest.raises(config.UnsupportedProvider, match="feedback"):
        config.models_for("feedback", "openai")


def test_openai_in_providers_and_models():
    assert "openai" in config.PROVIDERS
    for prompt in ("grammar", "qna"):
        assert config.MODELS_BY_PROMPT[prompt]["openai"] == ["gpt-5-nano"]
    assert "openai" not in config.MODELS_BY_PROMPT["feedback"]


@pytest.mark.parametrize("provider, expected", [
    ("openai", 32768), ("ollama", config.NUM_PREDICT), ("lmstudio", config.NUM_PREDICT),
])
def test_num_predict_for(provider, expected):
    assert config.num_predict_for(provider) == expected


def test_invoke_kwargs_openai_has_no_sampling_params():
    kw = _invoke_kwargs("openai", {"reasoning_effort": "low"}, 1000, 32768)
    assert kw == {"reasoning_effort": "low", "max_tokens": 32768}
    for key in ("seed", "temperature", "top_p"):
        assert key not in kw


@pytest.mark.parametrize("provider, params, seed, expected", [
    ("ollama", {"temperature": 0.2}, 7,
     {"options": {"temperature": 0.2, "num_ctx": config.NUM_CTX, "num_predict": 500, "seed": 7}}),
    ("ollama", {"temperature": 0.2}, None,
     {"options": {"temperature": 0.2, "num_ctx": config.NUM_CTX, "num_predict": 500}}),
    ("lmstudio", {"temperature": 0.2}, 7,
     {"temperature": 0.2, "max_tokens": 500, "seed": 7}),
    ("lmstudio", {"temperature": 0.2}, None,
     {"temperature": 0.2, "max_tokens": 500}),
])
def test_invoke_kwargs_local_unchanged(provider, params, seed, expected):
    assert _invoke_kwargs(provider, params, seed, 500) == expected


def test_openai_payload_offline():
    llm = ChatOpenAI(model="gpt-5-nano", api_key="test", temperature=0.1, max_tokens=50)
    kw = _invoke_kwargs("openai", {"reasoning_effort": "medium"}, None, 32768)
    payload = llm._get_request_payload("hi", **kw)
    assert payload["max_completion_tokens"] == 32768
    assert payload["reasoning_effort"] == "medium"
    assert "temperature" not in payload
    assert "max_tokens" not in payload


def _grouped() -> pd.DataFrame:
    rows = []
    for model, sid, bleu, wer in [
        ("local-a", "t0.0", 0.50, 0.30), ("local-a", "t0.2", 0.60, 0.20),
        ("local-b", "t0.0", 0.40, 0.40), ("local-b", "t0.2", 0.45, 0.35),
        ("gpt-5-nano", "effort_low", 0.55, 0.25), ("gpt-5-nano", "effort_medium", 0.70, 0.10),
    ]:
        temp = float(sid[1:]) if sid.startswith("t") else math.nan
        rows += [("bleu", sid, temp, model, bleu), ("wer", sid, temp, model, wer)]
    return pd.DataFrame(rows, columns=["metric", "setting_id", "temperature", "model", "mean"])


def test_load_grouped_keeps_effort_rows(tmp_path):
    long = _grouped().drop(columns="temperature")
    long["run"] = 0
    (tmp_path / "summary_long.csv").write_text(long.to_csv(index=False))

    class Paths:
        variant = "test"

        def result(self, name):
            return tmp_path / name

    out = ranking_rag_norag.load_grouped(Paths())
    assert out is not None
    effort = out[out["model"] == "gpt-5-nano"]
    assert set(effort["setting_id"]) == {"effort_low", "effort_medium"}
    assert effort["temperature"].isna().all()
    local = out[out["model"] == "local-a"]
    assert sorted(local["temperature"].unique()) == [0.0, 0.2]


def test_composite_keeps_effort_setting_ids():
    comp = ranking_rag_norag.composite(_grouped())
    flat = comp.reset_index() if "setting_id" not in comp.columns else comp
    nano = flat[flat["model"] == "gpt-5-nano"]
    assert set(nano["setting_id"]) == {"effort_low", "effort_medium"}
    assert {"model", "setting_id", "passed_gate", "score"} <= set(flat.columns)


def test_metric_ranking_best_temp():
    rank = ranking_rag_norag.metric_ranking(_grouped(), "bleu").set_index("model")
    assert rank.loc["gpt-5-nano", "best_temp"] == "effort_medium"
    assert rank.loc["local-a", "best_temp"] == "0.2"
