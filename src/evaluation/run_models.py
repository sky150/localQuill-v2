"""Model performance test: run every model x setting x run x text through the app's task and
model factory (src.llm.factory.get_model), see generation.py.

    uv run python -m src.evaluation.run_models --style formal --prompt grammar
    uv run python -m src.evaluation.run_models --style fiction --prompt grammar --models gemma4:e2b --runs 3
    uv run python -m src.evaluation.run_models --style formal --prompt grammar --provider lmstudio

The provider is "ollama" or "lmstudio" (config.PROVIDER by default, --provider to override);
config.MODELS_BY_PROVIDER lists the default models of each. Appends one row per generation to
the runs file of the chosen style and prompt (EvalPaths.runs_file) and resumes where it
stopped. All evaluators read only that file, so a vLLM runner that writes the same fields
(backend="vllm") plugs in without changes.

Row fields: backend (the provider), model, setting_id, params, run, seed, text_id, prompt,
style, output, raw_output, wall_s, load_s, prompt_tokens, gen_tokens, gen_tok_per_s,
done_reason. For LM Studio, run_models first (re)loads each model with config.NUM_CTX (LLM_CONTEXT).
Rows from LM Studio have load_s and gen_tok_per_s null: its API reports neither.
"""

from __future__ import annotations

import argparse

from . import config
from .common import (add_target_args, append_jsonl, build_messages, clean_output, load_testset,
                     paths_from_args, read_jsonl, run_key)
from .generation import build_task, ensure_context, generate, warm_up


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", choices=config.PROVIDERS, default=config.PROVIDER)
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--runs", type=int, default=config.N_RUNS)
    add_target_args(ap)
    args = ap.parse_args()
    paths = paths_from_args(args)
    models = args.models or config.MODELS_BY_PROVIDER[args.provider]

    items = load_testset(paths)
    done = {run_key(r) for r in read_jsonl(paths.runs_file)}
    total = len(models) * sum(config.runs_for(s, args.runs) for s in config.SETTINGS) * len(items)
    n = 0

    for model in models:
        ensure_context(args.provider, model)
        task = build_task(paths, args.provider, model)
        warm_up(task, args.provider)
        for setting in config.SETTINGS:
            params = {k: v for k, v in setting.items() if k != "id"}
            for run in range(config.runs_for(setting, args.runs)):
                seed = config.SEED_BASE + run
                for item in items:
                    n += 1
                    key = (model, setting["id"], run, item["id"])
                    if key in done:
                        continue
                    prompt = build_messages(paths, item)[0]["content"]
                    gen = generate(task, args.provider, prompt, params, seed)
                    append_jsonl(paths.runs_file, {
                        "backend": args.provider, "model": model, "setting_id": setting["id"],
                        "params": params, "run": run, "seed": seed,
                        "text_id": item["id"], "prompt": paths.prompt,
                        "style": paths.style,
                        "output": clean_output(gen.raw), "raw_output": gen.raw,
                        "wall_s": gen.wall_s, "load_s": gen.load_s,
                        "prompt_tokens": gen.prompt_tokens,
                        "gen_tokens": gen.gen_tokens,
                        "gen_tok_per_s": gen.gen_tok_per_s,
                        "done_reason": gen.done_reason,
                    })
                    print(f"[{n}/{total}] {model} {setting['id']} run{run} {item['id']} "
                          f"{gen.wall_s:.1f}s")


if __name__ == "__main__":
    main()
