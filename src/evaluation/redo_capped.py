"""Regenerate the grammar rows that hit the old output cap, and check the result.

Ollama reproduces a generation from seed, model, num_ctx and prompt, and the cap only acts once
reached. A row that stopped by itself under the old cap comes out the same under a higher one, so
redoing only the cut-off rows equals a full rerun; `--report` checks this by comparing each new
output with the archived one.

    uv run python -m src.evaluation.redo_capped --style fiction --prompt grammar --dry-run
    uv run python -m src.evaluation.redo_capped --style fiction --prompt grammar
    uv run python -m src.evaluation.run_all ... --skip build_references   (regenerate and rescore)
    uv run python -m src.evaluation.redo_capped --style fiction --prompt grammar --report

The redo step moves the stale rows to capped_rows.jsonl and removes them from runs.jsonl and
judge.jsonl, because eval_judge resumes per key and would keep scores of the old output.
eval_errant and eval_embedding recompute every row of runs.jsonl on each run, so they need no
cleanup.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from . import config
from .common import (EvalPaths, add_target_args, append_jsonl, normalize_for_scoring, paths_from_args,
                     read_jsonl, run_key)

CAPPED_FILE = "capped_rows.jsonl"


class RedoUnsupported(SystemExit):
    """--prompt is not grammar: only grammar had cut-off rows that can be redone alone."""


def check_prompt(prompt: str) -> None:
    if prompt != "grammar":
        raise RedoUnsupported(
            f"Redo is only supported for the grammar prompt, not {prompt!r}. Q&A follow-ups carry "
            "the parent answer, so a partial rerun would mix old and new conversations; "
            "feedback had no capped rows.")


# ----------------------------------------------------------------------------- pure logic


def stale_keys(rows: list[dict], cap_for: Callable[[str], int]) -> set[tuple]:
    """Keys of rows cut off under a cap lower than the one now configured for their backend.
    A row cut off at the current cap stays: a rerun would cut it again."""
    return {run_key(r) for r in rows
            if r.get("done_reason") == "length"
            and config.row_num_predict(r) < cap_for(r["backend"])}


def drop_keys(rows: list[dict], keys: set[tuple]) -> list[dict]:
    return [r for r in rows if run_key(r) not in keys]


@dataclass(frozen=True)
class RedoReport:
    n_archived: int
    n_regenerated: int
    n_missing: int
    n_prefix_match: int
    n_still_capped: int
    per_model: dict[str, tuple[int, int]]  # model -> (regenerated, still_capped)


def compare(archived: list[dict], current: list[dict]) -> RedoReport:
    """archived: the capped rows removed by redo. current: the rows of runs.jsonl now."""
    now = {run_key(r): r for r in current}
    regenerated = prefix = still = 0
    per_model: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for old in archived:
        new = now.get(run_key(old))
        if new is None:
            continue
        regenerated += 1
        per_model[old["model"]][0] += 1
        # The old output is a cut-off prefix of what a deterministic rerun produces.
        if normalize_for_scoring(new["output"]).startswith(normalize_for_scoring(old["output"])):
            prefix += 1
        if new.get("done_reason") == "length":
            still += 1
            per_model[old["model"]][1] += 1
    return RedoReport(len(archived), regenerated, len(archived) - regenerated, prefix, still,
                      {m: (a, b) for m, (a, b) in per_model.items()})


# ----------------------------------------------------------------------------- file steps


def _rewrite(path: Path, rows: list[dict]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(path)  # a crash leaves the old file, never a half-written one


def redo(paths: EvalPaths, models: set[str] | None, dry_run: bool) -> dict[str, int]:
    """Archive and remove the stale rows. Returns the stale count per model."""
    runs = read_jsonl(paths.runs_file)
    keys = stale_keys(runs, config.num_predict_for)
    stale = [r for r in runs if run_key(r) in keys and (not models or r["model"] in models)]
    keys = {run_key(r) for r in stale}
    counts: dict[str, int] = defaultdict(int)
    for r in stale:
        counts[r["model"]] += 1
    for m, n in sorted(counts.items()):
        print(f"{m}: {n}")
    print(f"{len(stale)} stale rows in {paths.runs_file}")
    if dry_run or not stale:
        return dict(counts)

    judge_file = paths.result("judge.jsonl")
    judge = read_jsonl(judge_file)
    # A redo that crashed before the rewrite leaves its rows archived; don't archive them twice.
    archived = {run_key(r) for r in read_jsonl(paths.result(CAPPED_FILE))}
    for r in stale:
        if run_key(r) not in archived:
            append_jsonl(paths.result(CAPPED_FILE), r)
    for src, name in ((paths.runs_file, "runs.before_redo.jsonl"),
                      (judge_file, "judge.before_redo.jsonl")):
        backup = paths.result(name)
        if src.exists() and not backup.exists():
            shutil.copy2(src, backup)
    _rewrite(paths.runs_file, drop_keys(runs, keys))
    if judge_file.exists():
        kept = drop_keys(judge, keys)
        _rewrite(judge_file, kept)
        print(f"removed {len(judge) - len(kept)} judge rows")
    return dict(counts)


def report(paths: EvalPaths) -> RedoReport:
    archive = paths.result(CAPPED_FILE)
    if not archive.exists():
        raise SystemExit(f"No {archive}; run the redo step first.")
    rep = compare(read_jsonl(archive), read_jsonl(paths.runs_file))
    print(f"archived {rep.n_archived}, regenerated {rep.n_regenerated}, missing {rep.n_missing}, "
          f"output extends archived {rep.n_prefix_match}, still capped {rep.n_still_capped}")
    print(f"{'model':32} {'regenerated':>11} {'still capped':>12}")
    for m, (a, b) in sorted(rep.per_model.items()):
        print(f"{m:32} {a:>11} {b:>12}")
    return rep


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_target_args(ap)
    ap.add_argument("--models", nargs="+", help="only redo these models")
    ap.add_argument("--dry-run", action="store_true", help="count stale rows, change nothing")
    ap.add_argument("--report", action="store_true", help="compare regenerated rows with the archive")
    args = ap.parse_args()
    check_prompt(args.prompt)
    paths = paths_from_args(args)
    if args.report:
        report(paths)
        return
    redo(paths, set(args.models) if args.models else None, args.dry_run)
    if not args.dry_run:
        flag = " --norag" if args.norag else ""
        print("\nNext:\n  uv run python -m src.evaluation.run_all --style "
              f"{args.style} --prompt {args.prompt}{flag} --skip build_references\n"
              f"  uv run python -m src.evaluation.redo_capped --style {args.style} "
              f"--prompt {args.prompt}{flag} --report")


if __name__ == "__main__":
    main()
