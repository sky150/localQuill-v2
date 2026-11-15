"""Retrieval-only check of the label-picking style query (src/tasks/base.py style_query).

The task model picks a problem label for each draft; the label is searched in the eval index
(nomic, 1000/400) as Task.search_guides does. Reports the rank of the known right passage, without
and with re-ranking (top 30 -> cross-encoder).

    uv run python tests/eval/rag/check_label_queries.py --ids st-f-04,st-f-05
"""

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE))

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="qwen3.5:4b")
parser.add_argument("--ids", default="")
parser.add_argument("--out", default="label_queries_check.json", help="file name in evaluation/reports/RAG/")
args = parser.parse_args()
os.environ.update({"GRAMMAR_MODEL": args.model, "FEEDBACK_MODEL": args.model, "LIMIT_TOKEN_NUMBER": "0"})

from langchain_chroma import Chroma  # noqa: E402
from sentence_transformers import CrossEncoder  # noqa: E402

from run_rag_eval import ROOT, covers, index_dir, load_raw, make_embeddings, snippet_spans  # noqa: E402
from src.config import get_config  # noqa: E402
from src.llm.factory import get_model  # noqa: E402
from src.tasks.feedback import FeedbackTask  # noqa: E402
from src.tasks.grammar import GrammarTask  # noqa: E402

TASKS = {"grammar": GrammarTask, "feedback": FeedbackTask}
OUT = ROOT / "evaluation" / "reports" / "RAG" / args.out


def main() -> int:
    emb = make_embeddings("nomic-embed-text", False)
    folder = str(index_dir("nomic-embed-text", 1000, 400, False))
    dbs = {c: Chroma(collection_name=c, embedding_function=emb, persist_directory=folder) for c in ("formal", "fiction")}
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    cases = [c for c in json.loads((HERE / "testset_style.json").read_text())["cases"]
             if not args.ids or c["id"] in args.ids.split(",")]
    rows = []
    for case in cases:
        config = get_config(case["task"], case["style"])
        task = TASKS[case["task"]](get_model(case["task"], case["style"], cfg=config), config, style=case["style"])
        label = task.style_query(case["draft"])
        db = dbs[case["style"]]
        plain = db.similarity_search(label, k=8)
        pool = db.similarity_search(label, k=30)
        scores = reranker.predict([(label, d.page_content) for d in pool])
        reranked = [d for d, _ in sorted(zip(pool, scores), key=lambda x: -x[1])][:8]

        def rank(docs):
            for i, d in enumerate(docs, 1):
                for g in case["gold"]:
                    if d.metadata["source"] == g["file"] and any(
                            covers(d.metadata["start_index"], len(d.page_content), s)
                            for s in snippet_spans(load_raw(ROOT / g["file"]), g["snippet"])):
                        return i
            return None

        row = {"id": case["id"], "task": case["task"], "issue": case["issue"], "picked": [label],
               "rank_vector": rank(plain), "rank_rerank": rank(reranked)}
        rows.append(row)
        print(f"{row['id']:<9} vector {str(row['rank_vector']):>4}  rerank {str(row['rank_rerank']):>4}  "
              f"| {row['issue']}  ->  {label}", flush=True)
    for key in ("rank_vector", "rank_rerank"):
        for task, k in (("feedback", 2), ("grammar", 3)):  # the app's top_k
            sub = [r for r in rows if r["task"] == task]
            if sub:
                print(f"{task} Hit@{k} {key[5:]}: {sum(1 for r in sub if r[key] and r[key] <= k)}/{len(sub)}")
    OUT.write_text(json.dumps(rows, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
