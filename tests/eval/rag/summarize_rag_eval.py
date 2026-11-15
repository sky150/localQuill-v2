"""Compare runs in evaluation/reports/RAG/results.jsonl as Markdown tables.

One table per task group (Q&A userdocs, grammar, feedback), one row per setup, sorted by Hit@k
at the k the app uses today, then MRR.

    uv run python tests/eval/rag/summarize_rag_eval.py                 # all runs
    uv run python tests/eval/rag/summarize_rag_eval.py --model nomic-embed-text --top 10
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from run_rag_eval import APP_TOP_K, REPORTS, metrics  # noqa: E402

# Task group -> (set, groups in results, k the app uses)
TABLES = {
    "Q&A (userdocs)": ("qa", {"userdocs"}, APP_TOP_K["userdocs"]),
    "Grammar (style guides)": ("style", {"grammar/formal", "grammar/fiction"}, APP_TOP_K["grammar"]),
    "Feedback (style guides)": ("style", {"feedback/formal", "feedback/fiction"}, APP_TOP_K["feedback"]),
}


def rows_for(runs, set_name, groups, k):
    rows = []
    for run in runs:
        by_style = {}
        for c in run["cases"]:
            if c["set"] == set_name and c["group"] in groups:
                by_style.setdefault(c["query_style"], []).append(c)
        for query_style, cases in by_style.items():
            m, m5 = metrics(cases, k), metrics(cases, 5)
            rows.append({
                "model": run["model"], "chunk": run["chunk_size"], "overlap": run["overlap"],
                "prefix": "on" if run["prefixes"] else "off", "query": query_style, "n": m["n"],
                "hit": m["hit"], "recall": m["recall"], "hit5": m5["hit"], "mrr": m["mrr"],
                "ms": run["query_ms_mean"],
            })
    return sorted(rows, key=lambda r: (-r["hit"], -r["mrr"]))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default=str(REPORTS / "results.jsonl"))
    parser.add_argument("--model")
    parser.add_argument("--top", type=int, default=0, help="rows per table, 0 = all")
    parser.add_argument("--exclude-query", default="issue", help="query styles to hide (issue = upper bound)")
    args = parser.parse_args()

    runs = [json.loads(line) for line in Path(args.file).read_text().splitlines() if line.strip()]
    if args.model:
        runs = [r for r in runs if r["model"] == args.model]
    hidden = set(filter(None, args.exclude_query.split(",")))

    for title, (set_name, groups, k) in TABLES.items():
        rows = [r for r in rows_for(runs, set_name, groups, k) if r["query"] not in hidden]
        if args.top:
            rows = rows[: args.top]
        print(f"\n### {title}: Hit@{k} (app k), Recall@{k}, Hit@5, MRR\n")
        print("| Model | Chunk | Overlap | Prefix | Query | Hit@k | Recall@k | Hit@5 | MRR | ms/query |")
        print("|---|---|---|---|---|---|---|---|---|---|")
        for r in rows:
            print(f"| {r['model']} | {r['chunk']} | {r['overlap']} | {r['prefix']} | {r['query']} "
                  f"| {r['hit']:.2f} | {r['recall']:.2f} | {r['hit5']:.2f} | {r['mrr']:.2f} | {r['ms']:.0f} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
