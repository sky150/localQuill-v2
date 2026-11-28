"""Tests for redo_capped (stale/drop/compare, prompt check, file round trip) and the config and
aggregate helpers it relies on. Offline: tmp_path files only."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from src.evaluation import config
from src.evaluation.aggregate import drop_excluded
from src.evaluation.redo_capped import (CAPPED_FILE, RedoReport, RedoUnsupported, check_prompt,
                                        compare, drop_keys, redo, stale_keys)

CAP = config.num_predict_for


def row(model="m1", run=0, text_id="t1", done="length", output="abc", backend="ollama",
        num_predict=..., setting_id="s1", **extra):
    r = {"model": model, "setting_id": setting_id, "run": run, "text_id": text_id,
         "backend": backend, "done_reason": done, "output": output}
    if num_predict is not ...:
        r["num_predict"] = num_predict
    return {**r, **extra}


def judge(model="m1", run=0, text_id="t1", criterion="c1", setting_id="s1"):
    return {"model": model, "setting_id": setting_id, "run": run, "text_id": text_id,
            "question_id": None, "criterion": criterion, "score": 3}


# ----------------------------------------------------------------------------- config


@pytest.mark.parametrize("r, expected", [
    ({}, 2048),
    ({"num_predict": None}, 2048),
    ({"num_predict": 4096}, 4096),
    ({"num_predict": 32768}, 32768),
])
def test_row_num_predict(r, expected):
    assert config.row_num_predict(r) == expected


def test_excluded_models():
    assert "qwen3.5:4b" in config.excluded_models("qna")
    assert config.excluded_models("grammar") == {}
    assert config.excluded_models("feedback") == {}


def test_drop_excluded_filters_only_excluded_prompt():
    df = pd.DataFrame({"model": ["qwen3.5:4b", "other"], "x": [1, 2]})
    assert list(drop_excluded(df, "qna")["model"]) == ["other"]
    assert list(drop_excluded(df, "grammar")["model"]) == ["qwen3.5:4b", "other"]


# ----------------------------------------------------------------------------- stale_keys


@pytest.mark.parametrize("r, stale", [
    (row(), True),                                          # legacy row, no num_predict
    (row(num_predict=2048), True),
    (row(num_predict=4096), False),                         # cut off at the current cap
    (row(done="stop"), False),
    (row(done="stop", num_predict=2048), False),
    (row(backend="openai", num_predict=32768), False),
    (row(backend="openai", num_predict=2048), True),
])
def test_stale_keys(r, stale):
    keys = stale_keys([r], CAP)
    assert (len(keys) == 1) is stale


def test_stale_keys_uses_backend_cap():
    assert stale_keys([row(num_predict=4096, backend="openai")], CAP)
    assert not stale_keys([row(num_predict=4096, backend="ollama")], CAP)


# ----------------------------------------------------------------------------- drop_keys


def test_drop_keys_keeps_order_and_removes_runs_rows():
    rows = [row(run=0, done="stop"), row(run=1), row(run=2, done="stop")]
    kept = drop_keys(rows, stale_keys(rows, CAP))
    assert [r["run"] for r in kept] == [0, 2]


def test_drop_keys_removes_all_judge_rows_of_a_key():
    j = [judge(run=0, criterion="a"), judge(run=0, criterion="b"), judge(run=1, criterion="a")]
    keys = stale_keys([row(run=0)], CAP)
    kept = drop_keys(j, keys)
    assert [(r["run"], r["criterion"]) for r in kept] == [(1, "a")]


def test_drop_keys_empty_keys_keeps_everything():
    rows = [row(run=0), row(run=1)]
    assert drop_keys(rows, set()) == rows


# ----------------------------------------------------------------------------- compare


def test_compare_counts():
    old = [row(model="a", run=0, output="Hello wor"),
           row(model="a", run=1, output="Hello"),
           row(model="b", run=0, output="xyz"),
           row(model="b", run=1, output="gone")]
    new = [row(model="a", run=0, output="Hello  world again", done="stop", num_predict=4096),
           row(model="a", run=1, output="Hello there", done="length", num_predict=4096),
           row(model="b", run=0, output="different", done="stop", num_predict=4096)]
    rep = compare(old, new)
    assert rep == RedoReport(n_archived=4, n_regenerated=3, n_missing=1, n_prefix_match=2,
                             n_still_capped=1, per_model={"a": (2, 1), "b": (1, 0)})


def test_compare_nothing_regenerated():
    rep = compare([row()], [])
    assert (rep.n_archived, rep.n_regenerated, rep.n_missing) == (1, 0, 1)
    assert rep.per_model == {}


# ----------------------------------------------------------------------------- check_prompt


@pytest.mark.parametrize("prompt", ["qna", "feedback", "other"])
def test_check_prompt_rejects(prompt):
    with pytest.raises(RedoUnsupported) as exc:
        check_prompt(prompt)
    assert isinstance(exc.value, SystemExit)
    assert prompt in str(exc.value)


def test_check_prompt_accepts_grammar():
    check_prompt("grammar")


# ----------------------------------------------------------------------------- redo on files


class StubPaths:
    def __init__(self, root):
        self.root = root
        self.runs_file = root / "runs.jsonl"

    def result(self, name):
        return self.root / name


def write(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def read(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


@pytest.fixture
def setup(tmp_path):
    runs = [row(model="a", run=0, text_id="t1", output="Hello wor"),               # stale
            row(model="a", run=1, text_id="t1", done="stop", output="Done"),
            row(model="b", run=0, text_id="t2", output="Cut o", num_predict=2048),  # stale
            row(model="b", run=1, text_id="t2", output="Still cut", num_predict=4096)]  # stays
    judges = [judge(model="a", run=0, text_id="t1", criterion=c) for c in "xy"]
    judges += [judge(model="a", run=1, text_id="t1"),
               judge(model="b", run=0, text_id="t2"),
               judge(model="b", run=1, text_id="t2")]
    p = StubPaths(tmp_path)
    write(p.runs_file, runs)
    write(p.result("judge.jsonl"), judges)
    return p, runs, judges


def test_redo_dry_run_changes_nothing(setup):
    p, runs, judges = setup
    counts = redo(p, None, dry_run=True)
    assert counts == {"a": 1, "b": 1}
    assert read(p.runs_file) == runs
    assert read(p.result("judge.jsonl")) == judges
    for name in (CAPPED_FILE, "runs.before_redo.jsonl", "judge.before_redo.jsonl"):
        assert not p.result(name).exists()


def test_redo_end_to_end(setup):
    p, runs, judges = setup
    assert redo(p, None, dry_run=False) == {"a": 1, "b": 1}

    archived = read(p.result(CAPPED_FILE))
    assert [(r["model"], r["run"]) for r in archived] == [("a", 0), ("b", 0)]
    assert [(r["model"], r["run"]) for r in read(p.runs_file)] == [("a", 1), ("b", 1)]
    assert [(r["model"], r["run"]) for r in read(p.result("judge.jsonl"))] == [("a", 1), ("b", 1)]
    assert read(p.result("runs.before_redo.jsonl")) == runs
    assert read(p.result("judge.before_redo.jsonl")) == judges
    assert not list(p.root.glob("*.tmp"))

    # Second redo: nothing stale left, archive and backups untouched.
    before = p.result(CAPPED_FILE).read_text(encoding="utf-8")
    assert redo(p, None, dry_run=False) == {}
    assert p.result(CAPPED_FILE).read_text(encoding="utf-8") == before
    assert read(p.result("runs.before_redo.jsonl")) == runs

    # Regenerated rows arrive; compare sees them.
    append = [row(model="a", run=0, text_id="t1", output="Hello world", done="stop",
                  num_predict=4096),
              row(model="b", run=0, text_id="t2", output="Cut off", done="length",
                  num_predict=4096)]
    write(p.runs_file, read(p.runs_file) + append)
    rep = compare(read(p.result(CAPPED_FILE)), read(p.runs_file))
    assert rep == RedoReport(2, 2, 0, 2, 1, {"a": (1, 0), "b": (1, 1)})


def test_redo_models_filter(setup):
    p, _, _ = setup
    assert redo(p, {"a"}, dry_run=False) == {"a": 1}
    assert [(r["model"], r["run"]) for r in read(p.result(CAPPED_FILE))] == [("a", 0)]
    assert ("b", 0) in [(r["model"], r["run"]) for r in read(p.runs_file)]


def test_redo_does_not_archive_a_key_twice(setup):
    p, _, _ = setup
    redo(p, None, dry_run=False)
    # Re-add a stale row, as after a crash between archiving and rewriting.
    write(p.runs_file, read(p.runs_file) + [row(model="a", run=0, text_id="t1", output="Hello wor")])
    assert redo(p, None, dry_run=False) == {"a": 1}
    keys = [(r["model"], r["run"], r["text_id"]) for r in read(p.result(CAPPED_FILE))]
    assert sorted(keys) == [("a", 0, "t1"), ("b", 0, "t2")]
    assert ("a", 0) not in [(r["model"], r["run"]) for r in read(p.runs_file)]
