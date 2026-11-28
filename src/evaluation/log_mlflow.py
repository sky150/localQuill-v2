"""Final step: publish the finished experiment to MLflow.

    uv run python -m src.evaluation.log_mlflow --style formal --prompt grammar

Run it after the scoring steps of the prompt (config.STEPS_BY_PROMPT): run_models, eval_errant,
eval_embedding and eval_judge for grammar; run_models and eval_judge for feedback and Q&A. The local
CSV/JSONL files in the results dir stay the cache between steps, because each step must
resume after an interruption and MLflow runs cannot be appended to cheaply.
This step only reads them and writes one MLflow run per (model, setting), shaped
like the runs of evaluation/notebooks/8_task_grammer_evaluation.ipynb so both
appear side by side. Every call creates new runs and never overwrites old ones.
"""

from __future__ import annotations

import argparse
import math
import re
from difflib import SequenceMatcher

import mlflow
import pandas as pd

from . import config
from .aggregate import summarize
from .common import (TASK_BY_PROMPT, EvalPaths, Question, add_target_args, build_messages, load_answers,
                     load_reference, load_runs, load_testset, normalize_for_scoring, paths_from_args,
                     read_jsonl, run_key, testset_sha)

KEYS = ["model", "setting_id", "run", "text_id"]
# Score files each prompt needs before it can be published (judge.jsonl for all of them).
SCORE_FILES = {"grammar": ("errant.csv", "embedding.csv", "judge.jsonl"),
               "feedback": ("judge.jsonl",), "qna": ("judge.jsonl",)}
ERRANT_COLS = ["tp_spans", "n_err_spans", "fp_union", "tp_best", "fp_best", "fn_best"]
EMBED_COLS = ["cos_doc", "cos_mean", "cos_min"]
_BAD_NAME = re.compile(r"[^A-Za-z0-9_\-. /]")


def clean(v):
    """Plain Python value, with NaN as None, so the table stays JSON-safe."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    return v.item() if hasattr(v, "item") else v


def metric_name(name: str) -> str:
    return _BAD_NAME.sub("_", name)


def similarity(response: str, expected: str | None) -> float | None:
    """The notebook's difflib ratio, on layout-neutral text.

    The notebook compares raw strings. Here both sides go through normalize_for_scoring
    first, because the references keep the source's hard line breaks and models reflow
    them, which would lower the ratio without any wording change.
    """
    if expected is None or not response.strip():
        return None
    return SequenceMatcher(None, normalize_for_scoring(response), normalize_for_scoring(expected)).ratio()


def prompt_for(paths: EvalPaths, item: dict, run: dict) -> str | None:
    # build_messages reads the frozen rag_cache; a missing entry must not abort the upload
    question = Question(run["question_id"], run["question"]) if run.get("question_id") else None
    try:
        return build_messages(paths, item, question, run.get("history"))[0]["content"]
    except Exception:
        return None


def expected_for(paths: EvalPaths, run: dict) -> str | None:
    """The reference answer: the opus correction (grammar), the answer-sheet facts (Q&A) or none
    (feedback has no reference)."""
    if paths.prompt == "grammar":
        return load_reference(paths, "opus", run["text_id"])
    if paths.prompt == "qna":
        facts = load_answers(paths, run["text_id"]).get(run["question_id"])
        return "\n".join(f"- {f}" for f in facts) if facts else None
    return None


def table_row(paths: EvalPaths, run: dict, item: dict, expected: str | None, prompt: str | None,
              extra: dict) -> dict:
    out = run["output"]
    row = {"input": item["text"], "expected": expected, "prompt": prompt, "response": out,
           "error": None if out.strip() else "empty output", "latency_s": run["wall_s"],
           "prompt_chars": len(prompt) if prompt is not None else None,
           "response_chars": len(out),
           # difflib against a reference text only makes sense for corrections
           "similarity": similarity(out, expected) if paths.prompt == "grammar" else None,
           "text_id": run["text_id"], "run": run["run"], "seed": run["seed"],
           "prompt_tokens": run.get("prompt_tokens"), "gen_tokens": run.get("gen_tokens"),
           "reasoning_tokens": run.get("reasoning_tokens"), "done_reason": run.get("done_reason"),
           "num_predict": config.row_num_predict(run)}
    if paths.prompt == "qna":
        row.update(question_id=run["question_id"], question=run["question"])
    return {**row, **{k: clean(v) for k, v in extra.items()}}


def load_extras(paths: EvalPaths) -> dict[tuple, dict]:
    """Per-generation errant, embedding and judge columns keyed by run_key."""
    keys = KEYS + (["question_id"] if paths.prompt == "qna" else [])
    judge = pd.DataFrame(read_jsonl(paths.results_dir / "judge.jsonl"))
    judge = judge.pivot_table(index=keys, columns="criterion", values="score", aggfunc="mean")
    judge.columns = [f"judge_{c}" for c in judge.columns]
    sources = [(judge.reset_index(), list(judge.columns))]
    if paths.prompt == "grammar":
        sources += [(pd.read_csv(paths.results_dir / "errant.csv"), ERRANT_COLS),
                    (pd.read_csv(paths.results_dir / "embedding.csv"), EMBED_COLS)]
    extras: dict[tuple, dict] = {}
    for df, cols in sources:
        cols = [c for c in cols if c in df.columns]
        for rec in df[keys + cols].to_dict("records"):
            extras.setdefault(run_key(rec), {}).update({c: rec[c] for c in cols})
    return extras


def log_group(paths: EvalPaths, model: str, setting_id: str, runs: list[dict], items: dict, per_run: pd.DataFrame,
              long: pd.DataFrame, extras: dict, sha: str) -> None:
    first = runs[0]
    with mlflow.start_run(run_name=f"{paths.prompt}_{paths.variant}_{model}_{setting_id}") as active:
        mlflow.set_tags({"testset_sha": sha, "backend": first["backend"],
                          "style": paths.style, "prompt": paths.prompt, "rag": str(paths.rag)})
        grammar = paths.prompt == "grammar"
        # Rows that stopped under the old cap equal a rerun under the new one, so the max is the
        # group's cap; the min shows a group whose rows came from both caps.
        caps = [config.row_num_predict(r) for r in runs]
        mlflow.log_params({
            "provider": first["backend"], "model": model,
            "temperature": first["params"].get("temperature"), "top_p": first["params"].get("top_p"),
            "reasoning_effort": first["params"].get("reasoning_effort"),
            "num_predict": max(caps), "num_predict_min": min(caps),
            "prompt": paths.prompt, "style": paths.style,
            "collection": paths.style,
            "rag": paths.rag, "top_k": config.RAG_TOP_K if paths.rag else 0, "n_cases": len({r["text_id"] for r in runs}),
            "n_runs": len({r["run"] for r in runs}), "num_ctx": config.NUM_CTX,
            "seed_base": config.SEED_BASE, "judge_model": config.JUDGE_MODEL_BY_PROMPT[paths.prompt],
            **({"embed_model": config.EMBED_MODEL, "references": ",".join(config.REFERENCES)} if grammar else {}),
            **({"n_questions": len({r["question_id"] for r in runs})} if paths.prompt == "qna" else {})})
        mlflow.log_text(TASK_BY_PROMPT[paths.prompt].template, "prompt_template.txt")

        rows, prompts = [], {}
        for i, r in enumerate(runs):
            item = items[r["text_id"]]
            # Grammar and feedback prompts are the same for every run of a text; a Q&A
            # follow-up carries this run's own earlier answer.
            pkey = (r["text_id"], r.get("question_id"), str(r.get("history")))
            if pkey not in prompts:
                prompts[pkey] = prompt_for(paths, item, r)
            row = table_row(paths, r, item, expected_for(paths, r), prompts[pkey],
                            extras.get(run_key(r), {}))
            rows.append(row)
            mlflow.log_metric("latency_s", row["latency_s"], step=i)
            if row["similarity"] is not None:
                mlflow.log_metric("similarity", row["similarity"], step=i)

        # Prefixed so the per-run series never shares a key with the summary mean below:
        # MLflow shows the value with the highest step, which would hide the mean.
        mine = per_run[(per_run.model == model) & (per_run.setting_id == setting_id)
                      ]
        for _, pr in mine.iterrows():
            for col, v in pr.drop(["model", "setting_id", "run"]).items():
                if pd.notna(v):
                    mlflow.log_metric(metric_name(f"per_run/{col}"), float(v), step=int(pr["run"]))

        lat = [r["latency_s"] for r in rows]
        sims = [r["similarity"] for r in rows if r["similarity"] is not None]
        summary = {"error_rate": sum(r["error"] is not None for r in rows) / len(rows),
                   "latency_mean_s": sum(lat) / len(lat), "latency_max_s": max(lat),
                   "response_chars_mean": sum(r["response_chars"] for r in rows) / len(rows)}
        if sims:
            summary["similarity_mean"] = sum(sims) / len(sims)
        for _, l in long[(long.model == model) & (long.setting_id == setting_id)].iterrows():
            summary[l["metric"]] = l["mean"]
            summary[f"{l['metric']}_std"] = l["std"]
        mlflow.log_metrics({metric_name(k): float(v) for k, v in summary.items() if pd.notna(v)})

        mlflow.log_table(data={k: [r[k] for r in rows] for k in rows[0]}, artifact_file="results.json")
        print(f"{model} / {setting_id}: run_id {active.info.run_id}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_target_args(ap)
    paths = paths_from_args(ap.parse_args())
    required = [paths.runs_file] + [paths.results_dir / f for f in SCORE_FILES[paths.prompt]]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise SystemExit("Missing result files, run that step first:\n  " + "\n  ".join(missing))

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT_BY_PROMPT[paths.prompt])

    per_run, long, _ = summarize(paths)
    items = {i["id"]: i for i in load_testset(paths)}
    sha = testset_sha(paths)
    extras = load_extras(paths)

    groups: dict[tuple, list[dict]] = {}
    for r in sorted(load_runs(paths), key=lambda r: (r["run"], r["text_id"])):
        groups.setdefault((r["model"], r["setting_id"]), []).append(r)
    excluded = config.excluded_models(paths.prompt)
    for model, reason in excluded.items():
        print(f"excluded {model}: {reason}")
    for (model, setting_id), runs in groups.items():
        if model in excluded:
            continue
        log_group(paths, model, setting_id, runs, items, per_run, long, extras, sha)


if __name__ == "__main__":
    main()
