"""Model performance test: run every model x setting x run x text through the app's task and
model factory (src.llm.factory.get_model), see generation.py.

    uv run python -m src.evaluation.run_models --style formal --prompt grammar
    uv run python -m src.evaluation.run_models --style fiction --prompt grammar --models gemma4:e2b --runs 3
    uv run python -m src.evaluation.run_models --style formal --prompt grammar --provider lmstudio
    uv run python -m src.evaluation.run_models --style formal --prompt grammar --provider openai
    uv run python -m src.evaluation.run_models --style formal --prompt grammar --norag
    uv run python -m src.evaluation.run_models --style formal --prompt feedback --texts Essay_C2_Profishency_Response_1

--norag runs the no-RAG baseline: the same task template with empty writing guidelines,
written to the "<style>-norag" results folder (EvalPaths.variant).

The provider is "ollama", "lmstudio" or "openai" (config.PROVIDER by default, --provider to
override); config.MODELS_BY_PROMPT lists the default models of each prompt and provider, and
config.settings_for the settings (SETTINGS_BY_PROMPT, or reasoning efforts for openai, which
needs OPENAI_API_KEY and supports grammar and qna only). Appends one row per generation to the
runs file of the chosen style and prompt (EvalPaths.runs_file) and resumes where it stopped. All evaluators read
only that file, so a vLLM runner that writes the same fields (backend="vllm") plugs in without
changes.

Row fields: backend (the provider), model, setting_id, params, run, seed, text_id, prompt,
style, rag, output, raw_output, wall_s, load_s, prompt_tokens, gen_tokens, gen_tok_per_s,
reasoning_tokens (null unless the provider reports hidden reasoning), done_reason, num_predict
(the output cap of the generation; older rows lack it, see config.row_num_predict). OpenAI rows
have seed null: the seed is not sent. Q&A rows add question_id, question and history (the
parent question and this model's answer to it, empty for a first question). For LM Studio, run_models first (re)loads
each model with config.NUM_CTX (LLM_CONTEXT). Rows from LM Studio have load_s and gen_tok_per_s
null: its API reports neither.

Q&A: every text gets each question of data/eval/prompts/qna/questions.json in file order, so a
follow-up always runs after its parent. Its history holds the parent answer of the same model,
setting and run, read back from runs.jsonl, so a resumed run continues the same conversation.
"""

from __future__ import annotations

import argparse

from . import config
from .common import (Question, add_target_args, append_jsonl, build_messages, clean_output,
                     load_questions, load_testset, paths_from_args, qna_history, read_jsonl, run_key)
from .generation import build_task, ensure_context, generate, warm_up


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", choices=config.PROVIDERS, default=config.PROVIDER)
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--runs", type=int, default=config.N_RUNS)
    ap.add_argument("--texts", nargs="+", default=None, help="only these text ids (default: all)")
    add_target_args(ap)
    args = ap.parse_args()
    paths = paths_from_args(args)
    models = args.models or config.models_for(paths.prompt, args.provider)
    settings = config.settings_for(paths.prompt, args.provider)

    items = load_testset(paths)
    if args.texts:
        unknown = set(args.texts) - {i["id"] for i in items}
        if unknown:
            raise SystemExit(f"Unknown text ids {sorted(unknown)} in {paths.input_texts}.")
        items = [i for i in items if i["id"] in args.texts]
    # None = one generation per text (grammar, feedback); Q&A asks each question per text.
    questions: list[Question | None] = load_questions(paths) if paths.prompt == "qna" else [None]
    by_id = {q.id: q for q in questions if q is not None}
    outputs = {run_key(r): r["output"] for r in read_jsonl(paths.runs_file)}
    per_model = sum(config.runs_for(s, args.runs) for s in settings) * len(items) * len(questions)
    total = len(models) * per_model
    n = 0

    for m_idx, model in enumerate(models, 1):
        m_n = 0
        print(f"=== model {m_idx}/{len(models)}: {model} ({per_model} generations) ===")
        ensure_context(args.provider, model)
        task = build_task(paths, args.provider, model)
        warm_up(task, args.provider)
        for setting in settings:
            params = {k: v for k, v in setting.items() if k != "id"}
            for run in range(config.runs_for(setting, args.runs)):
                # OpenAI gets no seed (generation.py does not send one), so the row records none.
                seed = None if args.provider == "openai" else config.SEED_BASE + run
                for item in items:
                    for question in questions:
                        n += 1
                        m_n += 1
                        qid = question.id if question else None
                        key = (model, setting["id"], run, item["id"], qid)
                        if key in outputs:
                            continue
                        row = {}
                        history = None
                        if question is not None:
                            parent = by_id.get(question.follow_up_of) if question.follow_up_of else None
                            parent_key = (model, setting["id"], run, item["id"], question.follow_up_of)
                            history = qna_history(question, parent, outputs.get(parent_key))
                            row = {"question_id": qid, "question": question.question, "history": history}
                        prompt = build_messages(paths, item, question, history)[0]["content"]
                        gen = generate(task, args.provider, prompt, params, seed)
                        output = clean_output(gen.raw)
                        outputs[key] = output
                        append_jsonl(paths.runs_file, {
                            "backend": args.provider, "model": model, "setting_id": setting["id"],
                            "params": params, "run": run, "seed": seed,
                            "text_id": item["id"], **row, "prompt": paths.prompt,
                            "style": paths.style, "rag": paths.rag,
                            "output": output, "raw_output": gen.raw,
                            "wall_s": gen.wall_s, "load_s": gen.load_s,
                            "prompt_tokens": gen.prompt_tokens,
                            "gen_tokens": gen.gen_tokens,
                            "gen_tok_per_s": gen.gen_tok_per_s,
                            "reasoning_tokens": gen.reasoning_tokens,
                            "done_reason": gen.done_reason,
                            "num_predict": config.num_predict_for(args.provider),
                        })
                        label = f"{item['id']}{f' {qid}' if qid else ''}"
                        print(f"[model {m_idx}/{len(models)} {m_n}/{per_model} | total {n}/{total}] "
                              f"{model} {setting['id']} run{run} {label} {gen.wall_s:.1f}s")


if __name__ == "__main__":
    main()
