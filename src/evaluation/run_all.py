"""Run the full evaluation for one prompt, every step except log_mlflow.

    uv run python -m src.evaluation.run_all                        # grammar, every style
    uv run python -m src.evaluation.run_all --style formal
    uv run python -m src.evaluation.run_all --style fiction --skip build_references
    uv run python -m src.evaluation.run_all --norag --skip build_references   # no-RAG baseline

Runs the steps of README.md "Run order" (1 to 4) in that order, one style after the other.
Each step runs in its own process with the same --style and --prompt, so its argparse
defaults apply. It stops at the first failing step. Every step resumes where it stopped, so a
rerun after a crash repeats no finished work. Publish to MLflow afterwards with log_mlflow.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time

from . import config

STEPS = ("build_references", "run_models", "eval_errant", "eval_embedding", "eval_judge", "aggregate")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--style", nargs="+", choices=config.STYLES, default=list(config.STYLES))
    ap.add_argument("--prompt", choices=config.PROMPTS, default="grammar")
    ap.add_argument("--skip", nargs="+", choices=STEPS, default=[])
    ap.add_argument("--norag", action="store_true", help="pass --norag to every step")
    args = ap.parse_args()
    steps = [s for s in STEPS if s not in args.skip]

    start = time.perf_counter()
    for style in args.style:
        for i, step in enumerate(steps, 1):
            print(f"\n##### [{style} {i}/{len(steps)}] {step} #####", flush=True)
            t0 = time.perf_counter()
            cmd = [sys.executable, "-m", f"src.evaluation.{step}", "--style", style, "--prompt", args.prompt]
            if args.norag:
                cmd.append("--norag")
            if subprocess.run(cmd).returncode != 0:
                raise SystemExit(f"{step} failed for style {style}; fix it and rerun, finished work is kept.")
            print(f"##### {step} done in {time.perf_counter() - t0:.0f}s #####", flush=True)
    print(f"\nAll steps done in {(time.perf_counter() - start) / 60:.1f} min. "
          f"Next: uv run python -m src.evaluation.log_mlflow --style <style> --prompt {args.prompt}"
          f"{' --norag' if args.norag else ''}")


if __name__ == "__main__":
    main()
