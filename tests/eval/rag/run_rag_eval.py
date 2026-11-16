"""Retrieval eval for the RAG setup: Hit@k, Recall@k and MRR, no LLM judge.

Builds a Chroma index per (embedding model, chunk size, overlap, prefixes), cached under
evaluation/cache/rag/, runs both test sets and appends one JSON line per run to
evaluation/reports/RAG/results.jsonl. Every gold snippet's rank is stored, so any k can be
computed later without re-running.

    uv run python tests/eval/rag/run_rag_eval.py                       # current app setup
    uv run python tests/eval/rag/run_rag_eval.py --chunks 500,1000,1500 --overlaps 0,200,400
    uv run python tests/eval/rag/run_rag_eval.py --models nomic-embed-text,embeddinggemma --prefixes off,on

A chunk matches a gold snippet when it covers at least half of it. Snippets are located in the
source text and compared with the chunk's character span, so a snippet cut by a chunk
boundary still counts for the chunk holding most of it.

Not part of this eval: the guardrail rescan of retrieved userdocs/outline chunks (it only drops
chunks that look like instructions, none of the test corpus does).
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from langchain_chroma import Chroma  # noqa: E402
from langchain_core.documents import Document  # noqa: E402
from langchain_core.embeddings import Embeddings  # noqa: E402
from langchain_ollama import OllamaEmbeddings  # noqa: E402
from langchain_text_splitters import RecursiveCharacterTextSplitter  # noqa: E402

from src.config import OLLAMA_URL  # noqa: E402
from src.tasks.feedback import FeedbackTask  # noqa: E402
from src.tasks.grammar import GrammarTask  # noqa: E402
from src.vector_db.load_documents import SUPPORTED_EXTENSIONS, load_file  # noqa: E402

HERE = Path(__file__).parent
CACHE = ROOT / "evaluation" / "cache" / "rag"
REPORTS = ROOT / "evaluation" / "reports" / "RAG"
K_MAX = 8
BATCH = 64

# How each task retrieves today (src/tasks/*.py). Used for the "task" query style and the summary.
TASK_QUERY = {"grammar": GrammarTask.retrieval_query, "feedback": FeedbackTask.retrieval_query}
APP_TOP_K = {"grammar": 3, "feedback": 2, "userdocs": 3, "outline": 1}

# Prefixes recommended by each model's authors. OllamaEmbeddings sends none.
PREFIXES = {
    "nomic-embed-text": ("search_query: ", "search_document: "),
    "embeddinggemma": ("task: search result | query: ", "title: none | text: "),
    "qwen3-embedding": (
        "Instruct: Given a question or a text, retrieve relevant passages\nQuery: ",
        "",
    ),
    "mxbai-embed-large": ("Represent this sentence for searching relevant passages: ", ""),
}

_QUOTES = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', "—": "-", "–": "-"})


# ---------- text and matching ----------

def normalize_with_map(text: str) -> tuple[str, list[int]]:
    """Normalised text (as in validate_testsets.py) plus each char's position in the original."""
    out, idx, space = [], [], False
    for i, ch in enumerate(text.translate(_QUOTES)):
        if ch.isspace():
            space = bool(out)
            continue
        if space:
            out.append(" ")
            idx.append(i)
            space = False
        low = ch.lower()
        out.append(low if len(low) == 1 else ch)
        idx.append(i)
    return "".join(out), idx


def snippet_spans(raw: str, snippet: str) -> list[tuple[int, int]]:
    """Every (start, end) of the snippet in the raw text."""
    norm, idx = normalize_with_map(raw)
    target, _ = normalize_with_map(snippet)
    return [(idx[m.start()], idx[m.end() - 1] + 1) for m in re.finditer(re.escape(target), norm)]


def covers(chunk_start: int, chunk_len: int, span: tuple[int, int]) -> bool:
    start, end = span
    overlap = min(chunk_start + chunk_len, end) - max(chunk_start, start)
    return overlap >= (end - start) / 2


# ---------- corpus ----------

def load_raw(path: Path) -> str:
    """Markitdown output, cached on disk (the big PDFs take minutes to convert)."""
    stat = path.stat()
    key = hashlib.sha1(f"{path.resolve()}:{stat.st_mtime_ns}:{stat.st_size}".encode()).hexdigest()[:16]
    cached = CACHE / "docs" / f"{key}.md"
    if cached.exists():
        return cached.read_text()
    text = load_file(str(path)).page_content
    cached.parent.mkdir(parents=True, exist_ok=True)
    cached.write_text(text)
    return text


def corpus_files(entry) -> list[Path]:
    """A corpus entry is a directory (as the app's style collections) or a list of files."""
    if isinstance(entry, str):
        folder = ROOT / entry
        return sorted(p for p in folder.iterdir() if p.suffix.lower() in SUPPORTED_EXTENSIONS)
    return [ROOT / p for p in entry]


def collections() -> dict[str, list[Path]]:
    corpus = {}
    for name in ("testset_qa.json", "testset_style.json"):
        for collection, entry in json.loads((HERE / name).read_text())["corpus"].items():
            corpus[collection] = corpus_files(entry)
    return corpus


# ---------- embeddings and index ----------

class PrefixedEmbeddings(Embeddings):
    def __init__(self, inner: Embeddings, query_prefix: str = "", doc_prefix: str = ""):
        self.inner, self.query_prefix, self.doc_prefix = inner, query_prefix, doc_prefix

    def embed_documents(self, texts):
        return self.inner.embed_documents([self.doc_prefix + t for t in texts])

    def embed_query(self, text):
        return self.inner.embed_query(self.query_prefix + text)


def prefixes_for(model: str) -> tuple[str, str]:
    base = model.split(":")[0]
    for name, pair in PREFIXES.items():
        if base == name or base.startswith(name + "-"):  # embeddinggemma-2 uses Gemma's prompts
            return pair
    return ("", "")


def make_embeddings(model: str, prefix: bool) -> Embeddings:
    inner = OllamaEmbeddings(model=model, base_url=OLLAMA_URL)
    return PrefixedEmbeddings(inner, *prefixes_for(model)) if prefix else inner


def index_dir(model: str, chunk: int, overlap: int, prefix: bool) -> Path:
    doc_prefix = prefixes_for(model)[1] if prefix else ""
    tag = "pdoc" if doc_prefix else "plain"  # query prefixes don't change the index
    return CACHE / "index" / f"{model.replace(':', '_').replace('/', '_')}__c{chunk}_o{overlap}__{tag}"


def build_index(folder: Path, emb: Embeddings, chunk: int, overlap: int, corpus) -> dict:
    """Embed every collection once; return per-collection chunk counts and build time."""
    done = folder / "done.json"
    if done.exists():
        return json.loads(done.read_text())
    shutil.rmtree(folder, ignore_errors=True)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk, chunk_overlap=overlap, add_start_index=True
    )  # same settings as src/vector_db/chunk_documents.py
    info: dict = {"chunks": {}}
    start = time.perf_counter()
    for collection, files in corpus.items():
        db = Chroma(collection_name=collection, embedding_function=emb, persist_directory=str(folder))
        count = 0
        for path in files:
            doc = Document(page_content=load_raw(path), metadata={"source": str(path.relative_to(ROOT))})
            chunks = splitter.split_documents([doc])
            for i in range(0, len(chunks), BATCH):
                db.add_documents(chunks[i : i + BATCH])
            count += len(chunks)
            print(f"    {collection}: {path.name} ({len(chunks)} chunks)", flush=True)
        info["chunks"][collection] = count
    info["index_seconds"] = round(time.perf_counter() - start, 1)
    done.write_text(json.dumps(info))
    return info


# ---------- queries ----------

def style_query(case: dict, query_style: str) -> str:
    keywords = TASK_QUERY[case["task"]]
    if query_style == "task":
        return f"{keywords}\n\n{case['draft'][:2000]}"  # what the app sends today
    if query_style == "draft":
        return case["draft"][:2000]
    if query_style == "keywords":
        return keywords
    if query_style == "llm":
        return llm_query(case["task"], case["draft"])
    if query_style == "issue":
        return case["issue"]  # the hand-written label: upper bound, not usable in the app
    raise ValueError(f"Unknown query style: {query_style}")


QUERY_MODEL = os.getenv("RAG_EVAL_QUERY_MODEL", "ministral-3:8b")  # 2b models echo the prompt
# The free-text query prompt the app used in run 5 (frozen here so the cached queries and results
# stay reproducible; the app now picks from a label list, src/tasks/base.py STYLE_QUERY_PROMPT).
QUERY_PROMPT = """You help look up advice in a writing style guide.
Read the draft and identify its main {focus} problem. Write a search query of at most 12 words
that names the problem in general writing terms. Do not mention the draft's topic.
Reply with the query only.

DRAFT:
\"\"\"
{draft}
\"\"\""""
FOCUS = {"grammar": GrammarTask.query_focus, "feedback": FeedbackTask.query_focus}


def llm_query(task: str, draft: str) -> str:
    """Search query written by a small LLM (one extra call per request in the app). Cached."""
    cache_file = CACHE / "llm_queries.json"
    cache = json.loads(cache_file.read_text()) if cache_file.exists() else {}
    key = hashlib.sha1(f"{QUERY_MODEL}|{task}|{QUERY_PROMPT}|{draft}".encode()).hexdigest()
    if key not in cache:
        from langchain_ollama import ChatOllama

        llm = ChatOllama(model=QUERY_MODEL, base_url=OLLAMA_URL, temperature=0, reasoning=False)
        cache[key] = llm.invoke(QUERY_PROMPT.format(focus=FOCUS[task], draft=draft[:2000])).content.strip().strip('*"').strip()
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(cache, indent=1))
    return cache[key]


def load_cases(query_styles: list[str]) -> list[dict]:
    """Flatten both test sets into (case, collection, query) jobs."""
    jobs = []
    qa = json.loads((HERE / "testset_qa.json").read_text())
    for case in qa["cases"]:
        jobs.append({"set": "qa", "id": case["id"], "group": case["collection"],
                     "query_style": "question", "query": case["query"], "gold": case["gold"]})
    style = json.loads((HERE / "testset_style.json").read_text())
    for case in style["cases"]:
        for qs in query_styles:
            jobs.append({"set": "style", "id": case["id"], "group": f"{case['task']}/{case['style']}",
                         "query_style": qs, "query": style_query(case, qs), "gold": case["gold"],
                         "collection": case["style"]})
    return jobs


def gold_collection(gold_file: str, corpus: dict[str, list[Path]], allowed: set[str]) -> str:
    for collection, files in corpus.items():
        if collection in allowed and any(str(p.relative_to(ROOT)) == gold_file for p in files):
            return collection
    raise ValueError(f"{gold_file} is in none of {allowed}")


# ---------- run ----------

def run_one(model, chunk, overlap, prefix, query_styles, corpus) -> dict:
    emb = make_embeddings(model, prefix)
    folder = index_dir(model, chunk, overlap, prefix)
    print(f"\n== {model} | chunk {chunk} / overlap {overlap} | prefixes {'on' if prefix else 'off'}")
    info = build_index(folder, emb, chunk, overlap, corpus)
    dbs = {c: Chroma(collection_name=c, embedding_function=emb, persist_directory=str(folder)) for c in corpus}
    spans = {}  # (file, snippet) -> spans in the raw text

    results, latencies = [], []
    for job in load_cases(query_styles):
        allowed = {"userdocs", "outline"} if job["group"] == "both" else {job.get("collection", job["group"])}
        golds = []
        for g in job["gold"]:
            key = (g["file"], g["snippet"])
            if key not in spans:
                spans[key] = snippet_spans(load_raw(ROOT / g["file"]), g["snippet"])
            golds.append({**g, "collection": gold_collection(g["file"], corpus, allowed)})

        retrieved = {}
        for collection in sorted({g["collection"] for g in golds}):
            t = time.perf_counter()
            retrieved[collection] = dbs[collection].similarity_search(job["query"], k=K_MAX)
            latencies.append(time.perf_counter() - t)

        ranks = []
        for g in golds:
            rank = next(
                (i for i, doc in enumerate(retrieved[g["collection"]], 1)
                 if doc.metadata["source"] == g["file"]
                 and any(covers(doc.metadata["start_index"], len(doc.page_content), s)
                         for s in spans[(g["file"], g["snippet"])])),
                None,
            )
            ranks.append(rank)
        results.append({k: job[k] for k in ("set", "id", "group", "query_style")} | {"ranks": ranks})

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": model, "chunk_size": chunk, "overlap": overlap, "prefixes": prefix,
        "k_max": K_MAX, "chunks": info["chunks"], "index_seconds": info["index_seconds"],
        "query_ms_mean": round(1000 * sum(latencies) / len(latencies), 1),
        "cases": results,
    }


# ---------- metrics ----------

def metrics(cases: list[dict], k: int) -> dict:
    """Hit@k (any gold), Recall@k (share of golds), Complete@k (all golds), MRR (up to k_max)."""
    n = len(cases)
    found = [[r is not None and r <= k for r in c["ranks"]] for c in cases]
    first = [min((r for r in c["ranks"] if r), default=None) for c in cases]
    return {
        "n": n,
        "hit": sum(any(f) for f in found) / n,
        "recall": sum(sum(f) / len(f) for f in found) / n,
        "complete": sum(all(f) for f in found) / n,
        "mrr": sum(1 / r for r in first if r) / n,
    }


def summary(run: dict) -> None:
    groups = {}
    for c in run["cases"]:
        groups.setdefault((c["set"], c["group"], c["query_style"]), []).append(c)
    print(f"  index {run['index_seconds']}s, chunks {run['chunks']}, query {run['query_ms_mean']} ms")
    print(f"  {'set':<6}{'group':<18}{'query':<10}{'n':>3}  {'app k':>5}  {'hit':>5}{'recall':>8}{'MRR':>6}"
          f"   hit@1..{K_MAX}")
    for (s, group, qs), cases in sorted(groups.items()):
        task = group.split("/")[0]
        k = APP_TOP_K.get(task, APP_TOP_K["userdocs"])
        m = metrics(cases, k)
        curve = " ".join(f"{metrics(cases, i)['hit']:.2f}" for i in range(1, K_MAX + 1))
        print(f"  {s:<6}{group:<18}{qs:<10}{m['n']:>3}  {k:>5}  {m['hit']:>5.2f}{m['recall']:>8.2f}"
              f"{m['mrr']:>6.2f}   {curve}")
    misses = [c["id"] for c in run["cases"] if c["query_style"] in ("question", "task") and not any(c["ranks"])]
    if misses:
        print(f"  not found in top {K_MAX} (app query): {', '.join(misses)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", default=os.getenv("EMBEDDING_MODEL", "nomic-embed-text"))
    parser.add_argument("--chunks", default=os.getenv("CHUNK_SIZE", "1000"))
    parser.add_argument("--overlaps", default=os.getenv("CHUNK_OVERLAP", "400"))
    parser.add_argument("--prefixes", default="off", help="off, on, or off,on")
    parser.add_argument("--query-styles", default="task,draft,keywords")
    parser.add_argument("--out", default=str(REPORTS / "results.jsonl"))
    args = parser.parse_args()

    corpus = collections()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for model in args.models.split(","):
        for chunk in map(int, args.chunks.split(",")):
            for overlap in map(int, args.overlaps.split(",")):
                if overlap >= chunk:
                    continue
                for prefix in [p == "on" for p in args.prefixes.split(",")]:
                    run = run_one(model, chunk, overlap, prefix, args.query_styles.split(","), corpus)
                    with out.open("a") as f:
                        f.write(json.dumps(run) + "\n")
                    summary(run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
