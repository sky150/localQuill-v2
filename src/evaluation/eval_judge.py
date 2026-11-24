"""LLM-as-judge test (G-Eval style) for grammar corrections, feedback and Q&A answers.

    uv run python -m src.evaluation.eval_judge --style formal --prompt grammar
    uv run python -m src.evaluation.eval_judge --style formal --prompt grammar --runs 0 1 2 --criteria preservation
    uv run python -m src.evaluation.eval_judge --style formal --prompt feedback
    uv run python -m src.evaluation.eval_judge --style fiction --prompt qna

The judge never sees the model name or the prompt, and must not be one of the models under
test for that prompt. What it does see depends on the prompt (JUDGES):
- grammar: the original text and the corrected text.
- feedback: the user's text and the feedback, as DISPRO1's deepeval G-Eval did (INPUT and
  ACTUAL_OUTPUT), judged by DISPRO1's judge (config.JUDGE_MODEL_BY_PROMPT).
- qna: the question, the earlier turn for a follow-up, the answer sheet, the attached document
  as the model saw it, the retrieved userdocs and outline, and the answer.

G-Eval steps:
1. For each criterion, the judge writes evaluation steps once. They are
   frozen in data/eval/prompts/<prompt>/judge_steps.json so every model is judged with the
   same rubric. Edit that file by hand if a step is poor; delete it to regenerate.
2. Each output gets an integer score per criterion. If the judge server returns logprobs, the
   score is the probability-weighted mean over the score tokens (as in the G-Eval paper,
   temperature 0, one call). Otherwise the judge is sampled JUDGE_SAMPLES times and the median
   is kept. judge.jsonl stores the score divided by JudgeSpec.divisor: 1-5 for grammar and Q&A,
   0-1 for feedback (0-10 / 10, deepeval's scale and DISPRO1's numbers).

Appends to judge.jsonl in the results dir and resumes where it stopped.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections.abc import Callable
from dataclasses import dataclass

from . import config
from .common import (EvalPaths, Question, add_target_args, append_jsonl, load_answers, load_runs,
                     load_testset, ollama_chat, paths_from_args, read_jsonl, retrieve_qna_docs)
from src.tasks.utility import clip


@dataclass(frozen=True)
class JudgeSpec:
    """How one prompt's outputs are judged.

    render(item, run, paths) returns the evaluated material, ending in a blank line, or None
    when the row cannot be judged yet (a Q&A question without an answer sheet).
    """
    criteria: dict[str, str]
    steps_intro: str
    subject: str            # what gets rated: "corrected text", "feedback", "answer"
    low: int
    high: int
    divisor: float          # stored score = raw score / divisor
    render: Callable[[dict, dict, EvalPaths], str | None]
    scale_note: str = ""


def _grammar_render(item: dict, run: dict, paths: EvalPaths) -> str:
    return (f"Original text:\n<<<\n{item['text']}\n>>>\n\n"
            f"Corrected text:\n<<<\n{run['output']}\n>>>\n\n")


def _feedback_render(item: dict, run: dict, paths: EvalPaths) -> str:
    return (f"User's text:\n<<<\n{item['text']}\n>>>\n\n"
            f"Feedback:\n<<<\n{run['output']}\n>>>\n\n")


def _qna_render(item: dict, run: dict, paths: EvalPaths) -> str | None:
    facts = load_answers(paths, item["id"]).get(run["question_id"])
    if not facts:
        return None
    question = Question(run["question_id"], run["question"])
    parts = [f"Question:\n<<<\n{run['question']}\n>>>\n\n"]
    if run.get("history"):
        turns = "\n".join(f"{m['role']}: {m['content']}" for m in run["history"])
        parts.append(f"Earlier conversation (the question follows up on it):\n<<<\n{turns}\n>>>\n\n")
    parts.append("Answer sheet (facts a correct answer states):\n" + "\n".join(f"- {f}" for f in facts) + "\n\n")
    parts.append(f"Attached document, as the assistant saw it:\n<<<\n"
                 f"{clip(item['text'], config.QNA_DOC_CHARS)}\n>>>\n\n")
    for kind, label, limit in (("userdocs", "Other user documents", config.QNA_USERDOCS_CHARS),
                               ("outline", "User outline", config.QNA_OUTLINE_CHARS)):
        docs = clip(retrieve_qna_docs(paths, item, question, kind), limit)
        if docs:
            parts.append(f"{label}, as the assistant saw them:\n<<<\n{docs}\n>>>\n\n")
    parts.append(f"Answer:\n<<<\n{run['output']}\n>>>\n\n")
    return "".join(parts)


JUDGES = {
    # Two axes instead of four criteria, to halve the judge calls. Paired by direction: correction
    # rewards changes, preservation penalizes them, so an under-correcting model and an
    # over-editing model get different profiles instead of the same mixed score.
    "grammar": JudgeSpec(
        criteria={
            "correction": "Correction quality: spelling, punctuation (especially commas), subject-verb "
                          "agreement and tense consistency are correct, and awkward or broken sentences "
                          "are rewritten to be clear and fluent.",
            "preservation": "Preservation: the corrected text says the same thing as the original, with "
                            "nothing added, removed or changed in content. Sentences that were already fine "
                            "are left alone, the author's voice and tone are kept, and names, invented words, "
                            "Japanese honorifics, paragraph breaks and formatting are unchanged.",
        },
        steps_intro="You will be given an original text and a corrected version of it, produced by a "
                    "grammar-correction assistant for fiction and essays.",
        subject="corrected text", low=1, high=5, divisor=1, render=_grammar_render),
    # DISPRO1's style and clarity G-Eval metrics (metrics_generation.py). Their steps sit in
    # prompts/feedback/judge_steps.json. DISPRO1's grammar metric is left out: the feedback task
    # no longer comments on grammar.
    "feedback": JudgeSpec(
        criteria={
            "style": "Style Feedback Quality: how well the feedback improves the writing style of the "
                     "user's text.",
            "clarity": "Clarity Feedback Quality: how well the feedback improves the clarity and "
                       "understandability of the user's text.",
        },
        steps_intro="You will be given a user's text and feedback on it, produced by a writing "
                    "assistant for fiction and essays.",
        subject="feedback", low=0, high=10, divisor=10, render=_feedback_render,
        scale_note=" A score of n on this scale is n/10 on the 0.0-1.0 scoring guide."),
    "qna": JudgeSpec(
        criteria={
            "correctness": "Correctness: the answer states the facts on the answer sheet for this "
                           "question and contradicts none of them. For a follow-up, it answers the "
                           "follow-up in the context of the earlier conversation.",
            "groundedness": "Groundedness: everything the answer claims about the user's writing comes "
                            "from the attached document, the other user documents or the outline. It "
                            "invents no characters, arguments, events or quotes, and says so when the "
                            "documents do not contain the answer.",
        },
        steps_intro="You will be given a question about a user's document, the document, an answer "
                    "sheet of facts a correct answer states, and the answer of a writing assistant.",
        subject="answer", low=1, high=5, divisor=1, render=_qna_render),
}

STEPS_SCHEMA = {"type": "object", "properties": {"steps": {"type": "array", "items": {"type": "string"}}},
                "required": ["steps"]}

_logprobs_supported: bool | None = None


def score_schema(spec: JudgeSpec) -> dict:
    return {"type": "object",
            "properties": {"score": {"type": "integer", "minimum": spec.low, "maximum": spec.high}},
            "required": ["score"]}


def judge(model: str, messages: list[dict], fmt: dict, temperature: float, logprobs: bool = False) -> dict:
    return ollama_chat(model, messages,
                       {"temperature": temperature, "num_ctx": config.JUDGE_NUM_CTX},
                       url=config.JUDGE_URL, fmt=fmt, logprobs=logprobs)


def load_steps(paths: EvalPaths, spec: JudgeSpec, model: str) -> dict[str, list[str]]:
    steps = json.loads(paths.judge_steps.read_text(encoding="utf-8")) if paths.judge_steps.exists() else {}
    for name, desc in spec.criteria.items():
        if name in steps:
            continue
        prompt = (
            f"{spec.steps_intro}\n\n"
            f"Evaluation criterion - {name}: {desc}\n\n"
            "Write 3 to 5 concise evaluation steps an evaluator should follow to rate the "
            f"{spec.subject} on this criterion from {spec.low} to {spec.high}. Return JSON: {{\"steps\": [...]}}"
        )
        res = judge(model, [{"role": "user", "content": prompt}], STEPS_SCHEMA, temperature=0)
        steps[name] = json.loads(res["message"]["content"])["steps"]
        print(f"[steps] {name}: {len(steps[name])} steps")
    paths.judge_steps.parent.mkdir(parents=True, exist_ok=True)
    paths.judge_steps.write_text(json.dumps(steps, ensure_ascii=False, indent=2), encoding="utf-8")
    return steps


def score_prompt(spec: JudgeSpec, name: str, steps: list[str], material: str) -> list[dict]:
    step_txt = "\n".join(f"{i}. {s}" for i, s in enumerate(steps, 1))
    user = (
        f"Evaluation criterion - {name}: {spec.criteria[name]}\n\n"
        f"Evaluation steps:\n{step_txt}\n\n"
        f"{material}"
        f"Rate the {spec.subject} on '{name}' from {spec.low} (very poor) to {spec.high} (excellent)."
        f"{spec.scale_note} Return JSON: {{\"score\": <{spec.low}-{spec.high}>}}"
    )
    return [{"role": "system", "content": "/no_think\nYou are a strict, consistent text evaluator."},
            {"role": "user", "content": user}]


def weighted_score(res: dict, low: int = 1, high: int = 5) -> float | None:
    """Expected score from the logprobs of the first score token, if present."""
    allowed = {str(i) for i in range(low, high + 1)}
    for tok in res.get("logprobs") or []:
        if not isinstance(tok, dict) or tok.get("token", "").strip() not in allowed:
            continue
        probs = {}
        for alt in tok.get("top_logprobs") or [tok]:
            t = alt.get("token", "").strip()
            if t in allowed:
                probs[int(t)] = probs.get(int(t), 0.0) + math.exp(alt["logprob"])
        total = sum(probs.values())
        return sum(k * p for k, p in probs.items()) / total if total else None
    return None


def score(spec: JudgeSpec, model: str, messages: list[dict]) -> tuple[float, str, list[int]]:
    """Raw score on spec's scale, the method used, and the integer scores the judge returned."""
    global _logprobs_supported
    fmt = score_schema(spec)
    if _logprobs_supported is not False:
        res = judge(model, messages, fmt, temperature=0, logprobs=True)
        content = json.loads(res["message"]["content"])["score"]
        w = weighted_score(res, spec.low, spec.high)
        _logprobs_supported = w is not None
        # A tokenizer may split "10" into "1" and "0", so the first token cannot tell 1 from 10.
        ambiguous = spec.high >= 10 and content == 10
        if w is not None and not ambiguous:
            return w, "logprob_weighted", [content]
        if w is None:
            print("[judge] no logprobs returned -> sampling fallback")
    samples = [json.loads(judge(model, messages, fmt, config.JUDGE_TEMPERATURE)["message"]["content"])["score"]
               for _ in range(config.JUDGE_SAMPLES)]
    return float(statistics.median(samples)), "sample_median", samples


def main() -> None:
    ap = argparse.ArgumentParser()
    add_target_args(ap)
    ap.add_argument("--runs", type=int, nargs="+", help="only judge these run indices")
    ap.add_argument("--criteria", nargs="+", help="only these criteria (default: all of the prompt)")
    args = ap.parse_args()
    paths = paths_from_args(args)
    spec = JUDGES[paths.prompt]
    criteria = args.criteria or list(spec.criteria)
    unknown = set(criteria) - set(spec.criteria)
    if unknown:
        raise SystemExit(f"Unknown criteria {sorted(unknown)} for {paths.prompt}; choose from {list(spec.criteria)}.")
    judge_model = config.JUDGE_MODEL_BY_PROMPT[paths.prompt]
    out_file = paths.result("judge.jsonl")

    under_test = {m for models in config.MODELS_BY_PROMPT[paths.prompt].values() for m in models}
    if judge_model in under_test or judge_model == "CHANGE_ME":
        raise SystemExit(f"Judge {judge_model!r} is under test for {paths.prompt}; "
                         "set config.JUDGE_MODEL_BY_PROMPT to another model.")

    items = {i["id"]: i for i in load_testset(paths)}
    steps = load_steps(paths, spec, judge_model)
    done = {(r["model"], r["setting_id"], r["run"], r["text_id"], r.get("question_id"), r["criterion"])
            for r in read_jsonl(out_file)}
    runs_by_model: dict[str, list[dict]] = {}
    for r in load_runs(paths):
        if args.runs is None or r["run"] in args.runs:
            runs_by_model.setdefault(r["model"], []).append(r)
    total = sum(len(rs) for rs in runs_by_model.values()) * len(criteria)
    n = 0
    unjudgeable: set[tuple] = set()

    for m_idx, (model, runs) in enumerate(runs_by_model.items(), 1):
        per_model = len(runs) * len(criteria)
        m_n = 0
        print(f"=== model {m_idx}/{len(runs_by_model)}: {model} ({per_model} judgements) ===")
        for run in runs:
            item = items[run["text_id"]]
            qid = run.get("question_id")
            material = None
            for name in criteria:
                n += 1
                m_n += 1
                key = (model, run["setting_id"], run["run"], run["text_id"], qid, name)
                if key in done:
                    continue
                material = material if material is not None else spec.render(item, run, paths)
                if material is None:
                    unjudgeable.add((run["text_id"], qid))
                    continue
                raw, method, samples = score(spec, judge_model, score_prompt(spec, name, steps[name], material))
                value = raw / spec.divisor
                append_jsonl(out_file, {"model": model, "setting_id": run["setting_id"],
                                        "run": run["run"], "text_id": item["id"],
                                        **({"question_id": qid} if qid is not None else {}),
                                        "style": item["style"], "criterion": name, "score": value,
                                        "method": method, "samples": samples, "judge_model": judge_model})
                label = f"{item['id']}{f' {qid}' if qid else ''}"
                print(f"[model {m_idx}/{len(runs_by_model)} {m_n}/{per_model} | total {n}/{total}] "
                      f"{model} {run['setting_id']} run{run['run']} {label} {name}={value:.2f}")
    if unjudgeable:
        print(f"\nSkipped {len(unjudgeable)} (text, question) pairs without an answer sheet entry:")
        for text_id, qid in sorted(unjudgeable):
            print(f"  {paths.qna_dir(text_id) / 'answers.json'}: {qid}")


if __name__ == "__main__":
    main()
