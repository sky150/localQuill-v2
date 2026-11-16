"""End-to-end check (Phase 5): do the grammar and feedback outputs handle the known problem?

Runs the real task on every draft in testset_style.json under three guideline variants and asks a
judge model one narrow question per output: was the case's known problem fixed (grammar) or
pointed out (feedback)? Scores: 2 = yes, 1 = partly, 0 = no.

    hybrid     hand-picked guidelines plus similarity search (the app now)
    fixed      hand-picked guidelines from src/tasks/guidelines/ only
    retrieval  similarity search on the style guide only, as before (eval index: nomic, 1000/400)
    none       no guidelines
    oracle     hand-picked guidelines plus the right passages: searched with the hand-written
               problem label and re-ranked (top 30 -> cross-encoder). Not possible in the app;
               shows what perfect retrieval would add.

    uv run python tests/eval/rag/run_task_eval.py                       # 3 repeats, all variants
    uv run python tests/eval/rag/run_task_eval.py --repeats 1 --variants fixed,none

Writes one JSON line per output to evaluation/reports/RAG/task_eval.jsonl and prints a table.
Task outputs are cached, so a new judge can re-score without re-running the tasks.
"""

import argparse
import hashlib
import inspect
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--model", default="qwen3.5:4b", help="task model (app default)")
parser.add_argument("--judge", default="ministral-3:8b")
parser.add_argument("--variants", default="hybrid,fixed,retrieval,none")
parser.add_argument("--repeats", type=int, default=3)
parser.add_argument("--tasks", default="grammar,feedback", help="only run these tasks")
parser.add_argument("--ids", default="", help="only these case ids, comma-separated")
parser.add_argument("--no-judge", action="store_true", help="skip the judge (feedback is graded by hand)")
parser.add_argument("--out", default=str(ROOT / "evaluation" / "reports" / "RAG" / "task_eval.jsonl"))
args = parser.parse_args()

# Override the dev settings in .env before src.config reads them (load_dotenv keeps these).
os.environ.update({"GRAMMAR_MODEL": args.model, "FEEDBACK_MODEL": args.model,
                   "LIMIT_TOKEN_NUMBER": "0", "GRAMMAR_MAX_WORDS": "300"})

from langchain_chroma import Chroma  # noqa: E402
from langchain_ollama import ChatOllama  # noqa: E402

import src.rag.retriever as retriever  # noqa: E402
from run_rag_eval import CACHE, index_dir, make_embeddings  # noqa: E402
from src.config import OLLAMA_URL, get_config  # noqa: E402
from src.llm.factory import get_model  # noqa: E402
import src.tasks.base as base  # noqa: E402
from src.tasks.base import Task  # noqa: E402
from src.tasks.feedback import FeedbackTask  # noqa: E402
from src.tasks.grammar import GrammarTask  # noqa: E402

TASKS = {"grammar": GrammarTask, "feedback": FeedbackTask}
RESPONSE_TIME = re.compile(r"\n\n---\nResponse time: [\d.]+s$")
NOTE = "\n\n---\nNote:"  # app note appended after the model's answer

JUDGE_PROMPT = {
    "grammar": """You check a grammar correction.
The ORIGINAL text has this known problem: {issue}.

ORIGINAL:
\"\"\"{draft}\"\"\"

CORRECTED:
\"\"\"{output}\"\"\"

Was the known problem fixed in the corrected text?
2 = every instance is fixed, 1 = some instances are fixed, 0 = not fixed or made worse.
Ignore other changes. Reply as JSON: {{"score": 0|1|2, "reason": "<one sentence>"}}""",
    "feedback": """You check writing feedback.
The TEXT has this known problem: {issue}.

TEXT:
\"\"\"{draft}\"\"\"

FEEDBACK:
\"\"\"{output}\"\"\"

Does the feedback point out the known problem (in any wording) and give a usable fix direction?
2 = clearly identifies it with a usable fix, 1 = only touches on it or vague, 0 = misses it.
Ignore other points in the feedback. Reply as JSON: {{"score": 0|1|2, "reason": "<one sentence>"}}""",
}


ORIGINAL_FIXED = Task.fixed_guidelines
ORIGINAL_RETRIEVE = retriever.retrieve
ORIGINAL_STYLE_QUERY = Task.style_query
CURRENT_CASE: dict = {}  # the case being run, for the oracle variant
_reranker = None


def oracle_retrieve(db, query: str, top_k: int) -> list:
    """Vector top 30, reordered by a cross-encoder (as in try_retrieval_options.py). No guardrail
    screen: the style guides are not user-supplied."""
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    pool = db.similarity_search(query, k=30)
    scores = _reranker.predict([(query, d.page_content) for d in pool])
    ranked = sorted(zip(pool, scores), key=lambda x: -x[1])[:top_k]
    return [(d, float(sc)) for d, sc in ranked]  # same (doc, score) shape as retriever.retrieve


def use_variant(variant: str) -> None:
    """Switch where grammar/feedback get their guidelines from."""
    searches = variant in ("hybrid", "retrieval", "oracle")
    Task.fixed_guidelines = ORIGINAL_FIXED if variant in ("hybrid", "fixed", "oracle") else (lambda self: "")
    # oracle: the hand-written problem label is the query, no LLM-written query
    Task.style_query = (lambda self, text: CURRENT_CASE["issue"]) if variant == "oracle" else ORIGINAL_STYLE_QUERY
    if searches:  # on the eval index built with the app's settings
        emb = make_embeddings("nomic-embed-text", prefix=False)
        folder = str(index_dir("nomic-embed-text", 1000, 400, prefix=False))
        dbs = {c: Chroma(collection_name=c, embedding_function=emb, persist_directory=folder)
               for c in ("formal", "fiction")}
        retriever.get_db = lambda collection, *a: dbs[collection]

    def retrieve(collection, query, top_k=3):
        # No outline in this test; only hybrid and retrieval may search the style guides.
        if collection == "outline" or not searches:
            raise ValueError("not used in this eval")
        if variant == "oracle":
            return oracle_retrieve(retriever.get_db(collection), query, top_k)
        return ORIGINAL_RETRIEVE(collection, query, top_k)

    retriever.retrieve = retrieve  # used by retrieve_from and retrieve_reranked (both tasks)


def prompt_version(task: Task) -> str:
    """Short hash of everything that builds the prompt (template, guidelines, search query code),
    so a change gets new outputs instead of cached ones."""
    code = inspect.getsource(type(task).build_prompt) + inspect.getsource(Task.retrieve_context) \
        + inspect.getsource(Task.style_query) + inspect.getsource(Task.search_guides) \
        + inspect.getsource(retriever.format_references) + inspect.getsource(retriever.retrieve_reranked) + base.STYLE_QUERY_PROMPT + "|".join(task.problem_labels)
    return hashlib.sha1((task.template + task.fixed_guidelines() + code).encode()).hexdigest()[:8]


def run_task(case: dict, variant: str, repeat: int, cache: dict) -> tuple[str, float, str]:
    task_cls = TASKS[case["task"]]
    config = get_config(case["task"], case["style"])
    task = task_cls(get_model(case["task"], case["style"], cfg=config), config, style=case["style"])
    CURRENT_CASE.update(case)
    version = prompt_version(task)
    key = f"{args.model}|{variant}|{case['id']}|{repeat}|{version}"
    if key not in cache:
        start = time.perf_counter()
        output = RESPONSE_TIME.sub("", task.run(case["draft"]))
        cache[key] = {"output": output, "seconds": round(time.perf_counter() - start, 1)}
    return cache[key]["output"], cache[key]["seconds"], version


def judge(case: dict, output: str, llm) -> tuple[int | None, str]:
    if not output.split(NOTE)[0].strip():
        return 0, "empty output"  # the judge scored empty outputs as fixed, so don't ask it
    prompt = JUDGE_PROMPT[case["task"]].format(issue=case["issue"], draft=case["draft"], output=output)
    reply = llm.invoke(prompt).content
    try:
        data = json.loads(reply[reply.index("{"): reply.rindex("}") + 1])
        return int(data["score"]), str(data.get("reason", ""))
    except (ValueError, KeyError, TypeError):
        return None, f"unparsable judge reply: {reply[:200]}"


def main() -> int:
    cases = [c for c in json.loads((HERE / "testset_style.json").read_text())["cases"]
             if c["task"] in args.tasks.split(",") and (not args.ids or c["id"] in args.ids.split(","))]
    cache_file = CACHE / "task_outputs.json"
    cache = json.loads(cache_file.read_text()) if cache_file.exists() else {}
    judge_llm = ChatOllama(model=args.judge, base_url=OLLAMA_URL, temperature=0, format="json")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Pass 1: all task outputs (cached). Pass 2: judging. Task model and judge don't fit in
    # 8 GB VRAM together, so interleaving them would reload a model on every call.
    variants = args.variants.split(",")
    for variant in variants:
        use_variant(variant)
        print(f"\n== generating: {variant}")
        for case in cases:
            for repeat in range(args.repeats):
                _, seconds, _ = run_task(case, variant, repeat, cache)
                cache_file.write_text(json.dumps(cache, indent=1))
                print(f"  {case['id']} #{repeat}: {seconds}s", flush=True)

    print("\n== judging")
    scores = defaultdict(list)
    for variant in variants:
        use_variant(variant)  # else the cache key uses the last variant's guidelines and misses
        for case in cases:
            for repeat in range(args.repeats):
                output, seconds, version = run_task(case, variant, repeat, cache)  # from cache
                score, reason = (None, "not judged (--no-judge)") if args.no_judge else judge(case, output, judge_llm)
                print(f"  {variant} {case['id']} #{repeat}: {score}  {reason[:90]}", flush=True)
                record = {"model": args.model, "judge": args.judge, "variant": variant, "id": case["id"],
                          "task": case["task"], "style": case["style"], "repeat": repeat, "prompt": version,
                          "score": score, "reason": reason, "seconds": seconds, "output": output}
                with out.open("a") as f:
                    f.write(json.dumps(record) + "\n")
                if score is not None:
                    scores[(variant, case["task"])].append(score)

    print(f"\nMean score (0-2), task model {args.model}, judge {args.judge}, {args.repeats} repeats")
    print(f"{'variant':<10}{'grammar':>9}{'feedback':>10}")
    for variant in args.variants.split(","):
        row = [scores[(variant, t)] for t in ("grammar", "feedback")]
        print(f"{variant:<10}" + "".join(f"{(sum(s) / len(s) if s else float('nan')):>9.2f} " for s in row))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
