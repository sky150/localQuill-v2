"""LLM-as-judge test (G-Eval style) for sentence rewrites.

    uv run python -m src.evaluation.eval_judge --style formal --prompt grammar
    uv run python -m src.evaluation.eval_judge --style formal --prompt grammar --runs 0 1 2 --criteria preservation

The judge sees ONLY the original text and the returned text: no model name,
no reference, no prompt. It must not be one of the tested models.

G-Eval steps:
1. For each criterion, the judge writes evaluation steps once. They are
   frozen in data/eval/prompts/<prompt>/judge_steps.json so every model is judged with the
   same rubric. Edit that file by hand if a step is poor; delete it to regenerate.
2. Each output is scored 1-5 per criterion. If the judge server returns
   logprobs, the score is the probability-weighted mean over the digits 1-5
   (as in the G-Eval paper, temperature 0, one call). Otherwise the judge is
   sampled JUDGE_SAMPLES times and the median is kept.

Appends to judge.jsonl in the results dir and resumes where it stopped.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics

from . import config
from .common import (EvalPaths, add_target_args, append_jsonl, load_runs, load_testset, ollama_chat,
                     paths_from_args, read_jsonl)

# Two axes instead of four criteria, to halve the judge calls. Paired by direction: correction
# rewards changes, preservation penalizes them, so an under-correcting model and an
# over-editing model get different profiles instead of the same mixed score.
CRITERIA = {
    "correction": "Correction quality: spelling, punctuation (especially commas), subject-verb "
                  "agreement and tense consistency are correct, and awkward or broken sentences "
                  "are rewritten to be clear and fluent.",
    "preservation": "Preservation: the corrected text says the same thing as the original, with "
                    "nothing added, removed or changed in content. Sentences that were already fine "
                    "are left alone, the author's voice and tone are kept, and names, invented words, "
                    "Japanese honorifics, paragraph breaks and formatting are unchanged.",
}

SCORE_SCHEMA = {"type": "object", "properties": {"score": {"type": "integer", "minimum": 1, "maximum": 5}},
                "required": ["score"]}
STEPS_SCHEMA = {"type": "object", "properties": {"steps": {"type": "array", "items": {"type": "string"}}},
                "required": ["steps"]}

_logprobs_supported: bool | None = None


def judge(messages: list[dict], fmt: dict, temperature: float, logprobs: bool = False) -> dict:
    return ollama_chat(config.JUDGE_MODEL, messages,
                       {"temperature": temperature, "num_ctx": config.JUDGE_NUM_CTX},
                       url=config.JUDGE_URL, fmt=fmt, logprobs=logprobs)


def load_steps(paths: EvalPaths) -> dict[str, list[str]]:
    steps = json.loads(paths.judge_steps.read_text(encoding="utf-8")) if paths.judge_steps.exists() else {}
    for name, desc in CRITERIA.items():
        if name in steps:
            continue
        prompt = (
            "You will be given an original text and a corrected version of it, produced by a "
            "grammar-correction assistant for fiction and essays.\n\n"
            f"Evaluation criterion - {name}: {desc}\n\n"
            "Write 3 to 5 concise evaluation steps an evaluator should follow to rate the "
            "corrected text on this criterion from 1 to 5. Return JSON: {\"steps\": [...]}"
        )
        res = judge([{"role": "user", "content": prompt}], STEPS_SCHEMA, temperature=0)
        steps[name] = json.loads(res["message"]["content"])["steps"]
        print(f"[steps] {name}: {len(steps[name])} steps")
    paths.judge_steps.parent.mkdir(parents=True, exist_ok=True)
    paths.judge_steps.write_text(json.dumps(steps, ensure_ascii=False, indent=2), encoding="utf-8")
    return steps


def score_prompt(name: str, steps: list[str], original: str, corrected: str) -> list[dict]:
    step_txt = "\n".join(f"{i}. {s}" for i, s in enumerate(steps, 1))
    user = (
        f"Evaluation criterion - {name}: {CRITERIA[name]}\n\n"
        f"Evaluation steps:\n{step_txt}\n\n"
        f"Original text:\n<<<\n{original}\n>>>\n\n"
        f"Corrected text:\n<<<\n{corrected}\n>>>\n\n"
        f"Rate the corrected text on '{name}' from 1 (very poor) to 5 (excellent). "
        "Return JSON: {\"score\": <1-5>}"
    )
    return [{"role": "system", "content": "/no_think\nYou are a strict, consistent text evaluator."},
            {"role": "user", "content": user}]


def weighted_score(res: dict) -> float | None:
    """Expected score from the logprobs of the first digit token, if present."""
    for tok in res.get("logprobs") or []:
        if not isinstance(tok, dict) or tok.get("token", "").strip() not in {"1", "2", "3", "4", "5"}:
            continue
        probs = {}
        for alt in tok.get("top_logprobs") or [tok]:
            t = alt.get("token", "").strip()
            if t in {"1", "2", "3", "4", "5"}:
                probs[int(t)] = probs.get(int(t), 0.0) + math.exp(alt["logprob"])
        total = sum(probs.values())
        return sum(k * p for k, p in probs.items()) / total if total else None
    return None


def score(messages: list[dict]) -> tuple[float, str, list[int]]:
    global _logprobs_supported
    if _logprobs_supported is not False:
        res = judge(messages, SCORE_SCHEMA, temperature=0, logprobs=True)
        w = weighted_score(res)
        _logprobs_supported = w is not None
        if w is not None:
            return w, "logprob_weighted", [json.loads(res["message"]["content"])["score"]]
        print("[judge] no logprobs returned -> sampling fallback")
    samples = [json.loads(judge(messages, SCORE_SCHEMA, config.JUDGE_TEMPERATURE)["message"]["content"])["score"]
               for _ in range(config.JUDGE_SAMPLES)]
    return float(statistics.median(samples)), "sample_median", samples


def main() -> None:
    ap = argparse.ArgumentParser()
    add_target_args(ap)
    ap.add_argument("--runs", type=int, nargs="+", help="only judge these run indices")
    ap.add_argument("--criteria", nargs="+", choices=list(CRITERIA), default=list(CRITERIA))
    args = ap.parse_args()
    paths = paths_from_args(args)
    out_file = paths.result("judge.jsonl")

    if any(config.JUDGE_MODEL in m for m in config.MODELS_BY_PROVIDER.values()) or config.JUDGE_MODEL == "CHANGE_ME":
        raise SystemExit("Set JUDGE_MODEL to a model that is not under test.")

    items = {i["id"]: i for i in load_testset(paths)}
    steps = load_steps(paths)
    done = {(r["model"], r["setting_id"], r["run"], r["text_id"], r["criterion"]) for r in read_jsonl(out_file)}
    runs = [r for r in load_runs(paths) if args.runs is None or r["run"] in args.runs]
    total = len(runs) * len(args.criteria)
    n = 0

    for run in runs:
        item = items[run["text_id"]]
        for name in args.criteria:
            n += 1
            key = (run["model"], run["setting_id"], run["run"], run["text_id"], name)
            if key in done:
                continue
            value, method, samples = score(score_prompt(name, steps[name], item["text"], run["output"]))
            append_jsonl(out_file, {"model": run["model"], "setting_id": run["setting_id"],
                                    "run": run["run"], "text_id": item["id"], "style": item["style"],
                                    "criterion": name, "score": value, "method": method, "samples": samples})
            print(f"[{n}/{total}] {run['model']} {run['setting_id']} run{run['run']} {item['id']} "
                  f"{name}={value:.2f}")


if __name__ == "__main__":
    main()
