"""Compare this year's feedback run with last year's DISPRO1 run, text by text.

    uv run python -m src.evaluation.compare_dispro1 --style formal --prompt feedback
    uv run python -m src.evaluation.compare_dispro1 --style formal --prompt feedback --model ministral-3:8b --temperature 0.1

Reads runs.jsonl and judge.jsonl of results/feedback/formal[-norag]/ and DISPRO1's records in
config.DISPRO1_RESULTS. Both sides use the same texts (data/eval/formal/raw/), judge
(llama3.1:8b) and the style and clarity G-Eval metrics on a 0-1 scale. Rows join on
file_name == text_id + ".txt", and DISPRO1 records count only for the same model and temperature.
Writes dispro1_comparison.csv into the results dir: one row per text plus an "ALL" row with
the means over texts.

Read the timing columns with care: DISPRO1's duration_seconds is the time of one evaluation
record, and tests/eval/run_generation_eval.py starts that clock after the feedback was
generated, so it may time the judge, not the model. wall_s is the generation alone.
DISPRO1 recorded no token counts.
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

from . import config
from .common import UnsupportedPrompt, add_target_args, paths_from_args, read_jsonl

METRICS = ("style", "clarity")


def then_table(records: list[dict], model: str, temperature: float) -> pd.DataFrame:
    """DISPRO1 means per text for one model and temperature: style, clarity, their mean,
    duration_s and the number of records."""
    rows = []
    for r in records:
        if r.get("llm_model") != model or float(r.get("temperature", -1)) != temperature:
            continue
        scores = {t["metric"]: t["score"] for t in r.get("per_test", [])}
        rows.append({"text_id": r["file_name"].removesuffix(".txt"),
                     **{m: scores.get(m) for m in METRICS}, "duration_s": r.get("duration_seconds")})
    if not rows:
        return pd.DataFrame(columns=["text_id", *METRICS, "mean", "duration_s", "n"]).set_index("text_id")
    df = pd.DataFrame(rows)
    out = df.groupby("text_id").agg(**{m: (m, "mean") for m in METRICS},
                                    duration_s=("duration_s", "mean"), n=("style", "size"))
    out.insert(len(METRICS), "mean", out[list(METRICS)].mean(axis=1))
    return out


def now_table(runs: list[dict], judge: list[dict], model: str) -> pd.DataFrame:
    """This year's means per text for one model: style, clarity, their mean, wall_s, tokens and runs."""
    runs_df = pd.DataFrame([r for r in runs if r["model"] == model])
    judge_df = pd.DataFrame([j for j in judge if j["model"] == model])
    if runs_df.empty:
        raise SystemExit(f"No feedback runs of {model} found. Run run_models first.")
    out = runs_df.groupby("text_id").agg(wall_s=("wall_s", "mean"), prompt_tokens=("prompt_tokens", "mean"),
                                         gen_tokens=("gen_tokens", "mean"), n=("run", "size"))
    if not judge_df.empty:
        scores = judge_df.pivot_table(index="text_id", columns="criterion", values="score", aggfunc="mean")
        scores = scores.reindex(columns=list(METRICS))
        scores["mean"] = scores.mean(axis=1)
        out = scores.join(out, how="right")
    return out


def compare(now: pd.DataFrame, then: pd.DataFrame) -> pd.DataFrame:
    """Side by side, _now and _then columns per metric, plus the change and an ALL row."""
    table = now.add_suffix("_now").join(then.add_suffix("_then"), how="outer")
    for m in (*METRICS, "mean"):
        if f"{m}_now" in table and f"{m}_then" in table:
            table[f"{m}_change"] = table[f"{m}_now"] - table[f"{m}_then"]
    table.loc["ALL"] = table.mean(numeric_only=True)
    return table


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_target_args(ap)
    ap.add_argument("--model", default=config.models_for("feedback")[0])
    ap.add_argument("--temperature", type=float, default=config.SETTINGS_BY_PROMPT["feedback"][0]["temperature"],
                    help="DISPRO1 records of this temperature count")
    args = ap.parse_args()
    paths = paths_from_args(args)
    if paths.prompt != "feedback":
        raise UnsupportedPrompt("compare_dispro1 compares feedback runs only; use --prompt feedback.")

    with config.DISPRO1_RESULTS.open(encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    then = then_table(records, args.model, args.temperature)
    now = now_table(read_jsonl(paths.runs_file), read_jsonl(paths.result("judge.jsonl")), args.model)
    table = compare(now, then)

    out = paths.result("dispro1_comparison.csv")
    table.to_csv(out, index_label="text_id")
    shown = [c for c in table.columns if c.split("_")[0] in (*METRICS, "mean", "wall", "duration", "gen")]
    with pd.option_context("display.width", 220, "display.max_columns", 30, "display.float_format", "{:.3f}".format):
        print(table[shown])
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
