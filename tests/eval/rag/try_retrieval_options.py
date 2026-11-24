"""Quick retrieval-only comparison of search options for grammar/feedback (style test set).

No answers are generated and no judge runs: for each option it checks at which rank the known
right passage comes back (same matching as run_rag_eval.py). Uses the indexes that
run_rag_eval.py already built, and the cached LLM-written queries, so it runs in about a minute
(plus a one-off ~90 MB download of the re-ranking model).

    RAG_EVAL_QUERY_MODEL=qwen3.5:4b uv run python tests/eval/rag/try_retrieval_options.py

Options:
    search    vector (app today), bm25 (keyword search), hybrid (both, merged by rank),
              and each of them re-ranked by a cross-encoder (top 30 -> reordered)
    query     keywords (app before), llm (LLM-written, app now), issue (hand-written label of
              the problem: the best a query could do, not usable in the app)
    chunks    chunk size / overlap of the existing indexes
"""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from langchain_chroma import Chroma  # noqa: E402
from rank_bm25 import BM25Okapi  # noqa: E402

from run_rag_eval import (ROOT, covers, gold_collection, index_dir, load_cases, load_raw,  # noqa: E402
                          make_embeddings, collections, snippet_spans)

POOL = 30  # candidates handed to the re-ranker
KS = (1, 2, 3, 5, 8)
CHUNKS = [(1000, 400), (500, 200), (500, 0), (1500, 200)]
QUERIES = ["keywords", "llm", "issue"]
RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"
OUT = ROOT / "evaluation" / "reports" / "RAG" / "retrieval_options.json"

tokenize = lambda text: re.findall(r"[a-z0-9]+", text.lower())  # noqa: E731


def rrf(*rankings, k=60):
    """Reciprocal rank fusion: merge ranked id lists without comparing their scores."""
    score = defaultdict(float)
    for ranking in rankings:
        for i, doc_id in enumerate(ranking):
            score[doc_id] += 1 / (k + i + 1)
    return sorted(score, key=score.get, reverse=True)


def main() -> int:
    from sentence_transformers import CrossEncoder

    reranker = CrossEncoder(RERANKER)
    corpus = collections()
    jobs = [j for j in load_cases(QUERIES) if j["set"] == "style"]
    spans, rows = {}, []

    for chunk, overlap in CHUNKS:
        folder = index_dir("nomic-embed-text", chunk, overlap, False)
        emb = make_embeddings("nomic-embed-text", False)
        store = {}
        for c in ("formal", "fiction"):
            db = Chroma(collection_name=c, embedding_function=emb, persist_directory=str(folder))
            got = db.get(include=["documents", "metadatas"])
            docs = {i: (t, m) for i, t, m in zip(got["ids"], got["documents"], got["metadatas"])}
            store[c] = (db, docs, list(docs), BM25Okapi([tokenize(docs[i][0]) for i in docs]))
        print(f"\n== chunks {chunk}/{overlap}: {', '.join(f'{c} {len(s[1])}' for c, s in store.items())}")

        for job in jobs:
            golds = []
            for g in job["gold"]:
                key = (g["file"], g["snippet"])
                if key not in spans:
                    spans[key] = snippet_spans(load_raw(ROOT / g["file"]), g["snippet"])
                golds.append({**g, "collection": gold_collection(g["file"], corpus, {job["collection"]})})
            db, docs, ids, bm25 = store[job["collection"]]

            vec_ids = [d.id for d in db.similarity_search(job["query"], k=POOL)]
            scores = bm25.get_scores(tokenize(job["query"]))
            bm_ids = [ids[i] for i in sorted(range(len(ids)), key=lambda i: -scores[i])[:POOL]]
            rankings = {"vector": vec_ids, "bm25": bm_ids, "hybrid": rrf(vec_ids, bm_ids)[:POOL]}
            for name in list(rankings):
                cand = rankings[name]
                ce = reranker.predict([(job["query"], docs[i][0]) for i in cand])
                rankings[name + "+rerank"] = [cand[i] for i in sorted(range(len(cand)), key=lambda i: -ce[i])]

            for search, ranking in rankings.items():
                ranks = []
                for g in golds:
                    ranks.append(next((r for r, i in enumerate(ranking, 1)
                                       if docs[i][1]["source"] == g["file"]
                                       and any(covers(docs[i][1]["start_index"], len(docs[i][0]), s)
                                               for s in spans[(g["file"], g["snippet"])])), None))
                rows.append({"chunks": f"{chunk}/{overlap}", "search": search, "query": job["query_style"],
                             "id": job["id"], "group": job["group"], "ranks": ranks})

    OUT.write_text(json.dumps(rows, indent=0))
    hit = lambda cs, k: sum(any(r and r <= k for r in c["ranks"]) for c in cs) / len(cs)  # noqa: E731
    for task, k_app in (("feedback", 2), ("grammar", 3)):
        print(f"\n### {task}: Hit@k (app uses k={k_app})")
        print(f"{'chunks':<10}{'query':<10}{'search':<16}" + "".join(f"{'@' + str(k):>6}" for k in KS))
        groups = defaultdict(list)
        for r in rows:
            if r["group"].startswith(task):
                groups[(r["chunks"], r["query"], r["search"])].append(r)
        for (ch, q, s), cs in groups.items():
            print(f"{ch:<10}{q:<10}{s:<16}" + "".join(f"{hit(cs, k):>6.2f}" for k in KS))
    print(f"\nPer-case ranks: {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
