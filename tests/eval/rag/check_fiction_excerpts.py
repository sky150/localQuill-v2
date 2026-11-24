"""Do the six genre excerpts in data/styles/fiction crowd guide passages out of the search?

Uses the eval index with the app's settings (nomic-embed-text, chunk 1000, overlap 400) and the
app's query for grammar/feedback. Writes evaluation/reports/RAG/fiction_excerpts_check.json.

    uv run python tests/eval/rag/check_fiction_excerpts.py
"""

import collections
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2]))

from langchain_chroma import Chroma  # noqa: E402

from run_rag_eval import APP_TOP_K, REPORTS, index_dir, make_embeddings, style_query  # noqa: E402

emb = make_embeddings("nomic-embed-text", prefix=False)
db = Chroma(collection_name="fiction", embedding_function=emb,
            persist_directory=str(index_dir("nomic-embed-text", 1000, 400, prefix=False)))
is_excerpt = lambda doc: doc.metadata["source"].endswith(".md")  # guides are PDFs

sources = collections.Counter(m["source"].split("/")[-1] for m in db.get(include=["metadatas"])["metadatas"])
cases = [c for c in json.loads((HERE / "testset_style.json").read_text())["cases"] if c["style"] == "fiction"]
rows = []
for case in cases:
    for query_style in ("task", "draft", "keywords"):
        top = db.similarity_search(style_query(case, query_style), k=3)
        rows.append({"id": case["id"], "task": case["task"], "query_style": query_style,
                     "top3": ["excerpt" if is_excerpt(d) else "guide" for d in top],
                     "app_k": APP_TOP_K[case["task"]]})

result = {"chunks_per_source": dict(sources.most_common()), "cases": rows}
(REPORTS / "fiction_excerpts_check.json").write_text(json.dumps(result, indent=1))
for r in rows:
    if r["query_style"] == "task":
        print(r["id"], r["task"], r["top3"])
