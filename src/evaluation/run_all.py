"""Run the full evaluation for one prompt, every step except log_mlflow.

    uv run python -m src.evaluation.run_all                        # grammar, every style
    uv run python -m src.evaluation.run_all --style formal
    uv run python -m src.evaluation.run_all --style fiction --skip build_references
    uv run python -m src.evaluation.run_all --norag --skip build_references   # no-RAG baseline
    uv run python -m src.evaluation.run_all --prompt feedback                 # formal only
    uv run python -m src.evaluation.run_all --prompt qna
    uv run python -m src.evaluation.run_all --prompt grammar --provider openai --skip build_references

Runs the steps of the prompt (config.STEPS_BY_PROMPT, README.md "Run order") in that order,
one style after the other. --style defaults to every style of the prompt
(config.STYLES_BY_PROMPT). Each step runs in its own process with the same --style and
--prompt, so its argparse defaults apply; only run_models gets --provider. It stops at the
first failing step. Every step resumes where it stopped, so a rerun after a crash repeats no
finished work. Publish to MLflow afterwards with log_mlflow.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time

from . import config

ALL_STEPS = tuple(dict.fromkeys(s for steps in config.STEPS_BY_PROMPT.values() for s in steps))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--style", nargs="+", choices=config.STYLES, default=None)
    ap.add_argument("--prompt", choices=config.PROMPTS, default="grammar")
    ap.add_argument("--skip", nargs="+", choices=ALL_STEPS, default=[])
    ap.add_argument("--norag", action="store_true", help="pass --norag to every step")
    ap.add_argument("--provider", choices=config.PROVIDERS, default=config.PROVIDER,
                    help="model provider; only run_models takes it")
    args = ap.parse_args()
    steps = [s for s in config.STEPS_BY_PROMPT[args.prompt] if s not in args.skip]
    args.style = args.style or list(config.STYLES_BY_PROMPT[args.prompt])
    unsupported = [s for s in args.style if s not in config.STYLES_BY_PROMPT[args.prompt]]
    if unsupported:
        raise SystemExit(f"{args.prompt} runs for styles {config.STYLES_BY_PROMPT[args.prompt]} only, "
                         f"not {unsupported}.")

    start = time.perf_counter()
    for style in args.style:
        for i, step in enumerate(steps, 1):
            print(f"\n##### [{style} {i}/{len(steps)}] {step} #####", flush=True)
            t0 = time.perf_counter()
            cmd = [sys.executable, "-m", f"src.evaluation.{step}", "--style", style, "--prompt", args.prompt]
            if args.norag:
                cmd.append("--norag")
            if step == "run_models":
                cmd += ["--provider", args.provider]
            if subprocess.run(cmd).returncode != 0:
                raise SystemExit(f"{step} failed for style {style}; fix it and rerun, finished work is kept.")
            print(f"##### {step} done in {time.perf_counter() - t0:.0f}s #####", flush=True)
    print(f"\nAll steps done in {(time.perf_counter() - start) / 60:.1f} min. "
          f"Next: uv run python -m src.evaluation.log_mlflow --style <style> --prompt {args.prompt}"
          f"{' --norag' if args.norag else ''}")


if __name__ == "__main__":
    main()
