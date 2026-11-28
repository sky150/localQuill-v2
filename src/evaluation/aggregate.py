"""Combine all test results of one style and prompt into one table: mean, median, std over the runs.

    uv run python -m src.evaluation.aggregate --style formal --prompt grammar

Step 1: one value per (model, setting, run). ERRANT counts are summed
        over texts first (micro-average), then turned into rates.
Step 2: mean / median / std of those per-run values across the N runs.

Writes summary_long.csv (every statistic) and summary.csv (means, one row per
model x setting) into the results dir of the chosen style and prompt. Missing
test files are skipped, so feedback and Q&A summaries hold judge and timing columns only.
Q&A also gets summary_by_question.csv (means per model x setting x question).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import argparse

from . import config
from .common import EvalPaths, add_target_args, paths_from_args, read_jsonl

KEYS = ["model", "setting_id", "run"]


def drop_excluded(df: pd.DataFrame, prompt: str) -> pd.DataFrame:
    """Rows of models the prompt leaves out of its comparison (config.EXCLUDED_MODELS).
    The raw files keep those rows; only the tables built here drop them."""
    excluded = config.excluded_models(prompt)
    return df[~df["model"].isin(excluded)] if excluded else df


def ratio(a: float, b: float) -> float:
    return a / b if b else np.nan


def errant_metrics(g: pd.DataFrame) -> pd.Series:
    s = g.sum(numeric_only=True)
    p = ratio(s["tp_best"], s["tp_best"] + s["fp_best"])
    r = ratio(s["tp_best"], s["tp_best"] + s["fn_best"])
    out = {
        "tp_rate": ratio(s["tp_spans"], s["n_err_spans"]),          # TP over total errors, <= 1
        "tp_rate_consensus": ratio(s.get("tp_cons", np.nan), s.get("n_cons", np.nan)),
        "precision_best": p,
        "f05_best": ratio(1.25 * p * r, 0.25 * p + r),
        "overcorr_per_100tok": ratio(100 * s["fp_union"], s["orig_tokens"]),
    }
    if g["n_errors"].notna().all():
        # vs. known injected count; can exceed 1 when ERRANT splits one error into several edits
        out["tp_rate_known"] = ratio(s["tp_spans"], s["n_errors"])
    return pd.Series(out)


def per_run_tables(paths: EvalPaths) -> list[pd.DataFrame]:
    out = []
    r = paths.results_dir

    if (r / "errant.csv").exists():
        df = pd.read_csv(r / "errant.csv")
        out.append(df.groupby(KEYS).apply(errant_metrics, include_groups=False))

    if (r / "embedding.csv").exists():
        df = pd.read_csv(r / "embedding.csv")
        out.append(df.groupby(KEYS)[["cos_doc", "cos_mean", "cos_min"]].mean())

    judge = read_jsonl(r / "judge.jsonl")
    if judge:
        df = pd.DataFrame(judge)
        t = df.pivot_table(index=KEYS, columns="criterion", values="score", aggfunc="mean")
        t.columns = [f"judge_{c}" for c in t.columns]
        out.append(t)

    runs = read_jsonl(paths.runs_file)
    if runs:
        df = pd.DataFrame(runs)
        # OpenAI rows carry null gen_tok_per_s; a column of only nulls would load as object dtype.
        df["gen_tok_per_s"] = pd.to_numeric(df["gen_tok_per_s"])
        # Rows written before the OpenAI runs have no reasoning_tokens field.
        df["reasoning_tokens"] = (pd.to_numeric(df["reasoning_tokens"]) if "reasoning_tokens" in df
                                  else float("nan"))
        # Share of generations cut off at the output cap (README "Output cap").
        df["capped"] = (df["done_reason"] == "length").astype(float)
        g = df.groupby(KEYS)
        out.append(pd.DataFrame({"wall_s": g["wall_s"].mean(), "wall_s_median": g["wall_s"].median(),
                                 "gen_tok_per_s": g["gen_tok_per_s"].mean(),
                                 "prompt_tokens": g["prompt_tokens"].mean(),
                                 "gen_tokens": g["gen_tokens"].mean(),
                                 "reasoning_tokens": g["reasoning_tokens"].mean(),
                                 "capped_rate": g["capped"].mean()}))
    return out


def by_question(paths: EvalPaths) -> pd.DataFrame | None:
    """Q&A only: judge and timing means per model x setting x question, so a follow-up reads
    apart from the first questions. Means over every run and text. None without Q&A rows."""
    runs = pd.DataFrame(read_jsonl(paths.runs_file))
    if runs.empty or "question_id" not in runs.columns:
        return None
    runs = drop_excluded(runs, paths.prompt)
    keys = ["model", "setting_id", "question_id"]
    tables = [runs.groupby(keys)[["wall_s", "gen_tokens"]].mean()]
    judge = pd.DataFrame(read_jsonl(paths.result("judge.jsonl")))
    if not judge.empty and "question_id" in judge.columns:
        judge = drop_excluded(judge, paths.prompt)
        t = judge.pivot_table(index=keys, columns="criterion", values="score", aggfunc="mean")
        t.columns = [f"judge_{c}" for c in t.columns]
        tables.insert(0, t)
    return pd.concat(tables, axis=1)


def summarize(paths: EvalPaths) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (per_run, long, wide) and write nothing. Exits when no results exist."""
    tables = per_run_tables(paths)
    if not tables:
        raise SystemExit("No results found. Run the eval scripts first.")
    per_run = drop_excluded(pd.concat(tables, axis=1).reset_index(), paths.prompt)

    metrics = [c for c in per_run.columns if c not in KEYS]
    long = (per_run.melt(id_vars=KEYS, value_vars=metrics, var_name="metric")
            .groupby(["model", "setting_id", "metric"])["value"]
            .agg(mean="mean", median="median", std="std", n_runs="count").reset_index())

    wide = long.pivot_table(index=["model", "setting_id"], columns="metric", values="mean")
    wide = wide[[m for m in metrics if m in wide.columns]]
    return per_run, long, wide


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_target_args(ap)
    paths = paths_from_args(ap.parse_args())
    for model, reason in config.excluded_models(paths.prompt).items():
        print(f"excluded {model}: {reason}")
    _, long, wide = summarize(paths)
    long.to_csv(paths.result("summary_long.csv"), index=False)
    wide.to_csv(paths.result("summary.csv"))

    sort_col = "tp_rate" if "tp_rate" in wide.columns else wide.columns[0]
    view = wide.sort_values(sort_col, ascending=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20, "display.float_format", "{:.3f}".format):
        print(view)
        per_question = by_question(paths)
        if per_question is not None:
            per_question.to_csv(paths.result("summary_by_question.csv"))
            print(f"\n{per_question}")
    print(f"\n-> {paths.result('summary.csv')} and summary_long.csv"
          f"{' and summary_by_question.csv' if per_question is not None else ''}")


if __name__ == "__main__":
    main()
