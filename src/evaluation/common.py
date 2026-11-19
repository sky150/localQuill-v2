"""Shared helpers: eval paths, test set, prompts, RAG retrieval, Ollama calls, I/O."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path

import requests

from src.tasks.grammar import GrammarTask

from . import config

# ----------------------------------------------------------------------------- paths


class UnsupportedPrompt(SystemExit):
    """--prompt names a folder that has no prompt builder or metrics yet."""


@dataclass(frozen=True)
class EvalPaths:
    """Every folder and file of one (style, prompt) evaluation.

    data/eval/<style>/                 error_texts/, opus/, languagetool/, n_errors.json, agreement.csv
    data/eval/prompts/<prompt>/        rag_cache.json, judge_steps.json
    data/eval/results/<prompt>/<variant>/  runs.jsonl and all eval outputs (gitignored)

    rag=False is the no-RAG baseline: the prompt gets no writing guides, and its results go to
    "<style>-norag" so its rows never share a run_key with the RAG rows.
    """
    style: str
    prompt: str
    rag: bool = True

    @property
    def variant(self) -> str:
        return self.style if self.rag else f"{self.style}-norag"

    @property
    def style_dir(self) -> Path:
        return config.EVAL_DATA_DIR / self.style

    @property
    def error_texts(self) -> Path:
        return self.style_dir / "error_texts"

    @property
    def n_errors_file(self) -> Path:
        return self.style_dir / "n_errors.json"

    @property
    def agreement_file(self) -> Path:
        return self.style_dir / "agreement.csv"

    def reference_dir(self, name: str) -> Path:
        return self.style_dir / name

    @property
    def prompt_dir(self) -> Path:
        return config.EVAL_DATA_DIR / "prompts" / self.prompt

    @property
    def rag_cache(self) -> Path:
        return self.prompt_dir / "rag_cache.json"

    @property
    def judge_steps(self) -> Path:
        return self.prompt_dir / "judge_steps.json"

    @property
    def results_dir(self) -> Path:
        return config.EVAL_DATA_DIR / "results" / self.prompt / self.variant

    @property
    def runs_file(self) -> Path:
        return self.results_dir / "runs.jsonl"

    def result(self, name: str) -> Path:
        return self.results_dir / name


def eval_paths(style: str, prompt: str, rag: bool = True) -> EvalPaths:
    if style not in config.STYLES:
        raise SystemExit(f"Unknown style {style!r}; choose one of {config.STYLES}.")
    if prompt not in config.SUPPORTED_PROMPTS:
        raise UnsupportedPrompt(f"Prompt {prompt!r} has no prompt builder or metrics yet; "
                                f"supported: {config.SUPPORTED_PROMPTS}.")
    return EvalPaths(style, prompt, rag)


def add_target_args(ap: argparse.ArgumentParser) -> None:
    """The arguments every eval script needs to find its folders."""
    ap.add_argument("--style", required=True, choices=config.STYLES)
    ap.add_argument("--prompt", required=True, choices=config.PROMPTS)
    ap.add_argument("--norag", action="store_true",
                    help="no-RAG baseline: prompt without writing guides, results in <style>-norag/")


def paths_from_args(args: argparse.Namespace) -> EvalPaths:
    return eval_paths(args.style, args.prompt, rag=not args.norag)


# ----------------------------------------------------------------------------- I/O


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_testset(paths: EvalPaths) -> list[dict]:
    """One item per error_texts/<id>.txt: {"id", "style", "text", "n_errors"}.

    n_errors comes from the optional n_errors.json ({id: count}); None if absent.
    """
    files = sorted(paths.error_texts.glob("*.txt"))
    if not files:
        raise SystemExit(f"No test texts in {paths.error_texts}")
    counts = (json.loads(paths.n_errors_file.read_text(encoding="utf-8"))
              if paths.n_errors_file.exists() else {})
    return [{"id": f.stem, "style": paths.style, "text": f.read_text(encoding="utf-8"),
             "n_errors": counts.get(f.stem)} for f in files]


def testset_sha(paths: EvalPaths) -> str:
    """First 12 hex chars of a hash over all test text names and contents."""
    h = hashlib.sha256()
    for f in sorted(paths.error_texts.glob("*.txt")):
        h.update(f.name.encode() + b"\0" + f.read_bytes() + b"\0")
    return h.hexdigest()[:12]


def load_reference(paths: EvalPaths, name: str, text_id: str) -> str | None:
    path = paths.reference_dir(name) / f"{text_id}.txt"
    return path.read_text(encoding="utf-8") if path.exists() else None


def load_runs(paths: EvalPaths) -> list[dict]:
    runs = read_jsonl(paths.runs_file)
    if not runs:
        raise SystemExit(f"No runs found in {paths.runs_file}. Run run_models first.")
    return runs


def run_key(row: dict) -> tuple:
    return (row["model"], row["setting_id"], row["run"], row["text_id"])


# ----------------------------------------------------------------------------- text


_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE = re.compile(r"^```[a-zA-Z]*\n(.*?)\n```$", re.DOTALL)
# The app prompt asks for a trailing "UNKNOWN WORDS:" list, which is not part of the corrected text.
_UNKNOWN_WORDS = re.compile(r"(?:^[ \t]*-{3,}[ \t]*\n)?^[^\n]*UNKNOWN WORDS:.*\Z",
                            re.DOTALL | re.IGNORECASE | re.MULTILINE)


def clean_output(text: str) -> str:
    """Reduce a model reply to the corrected text alone.

    Removes thinking blocks, a wrapping code fence, a trailing "UNKNOWN WORDS:" section
    (with an optional "---" line before it) and one pair of wrapping triple double quotes,
    which the app template uses to delimit the text. Keeps everything else.
    """
    text = _THINK.sub("", text).strip()
    m = _FENCE.match(text)
    text = m.group(1).strip() if m else text
    text = _UNKNOWN_WORDS.sub("", text).strip()
    if len(text) >= 6 and text.startswith('"""') and text.endswith('"""'):
        text = text[3:-3]
    return text.strip()


_WS = re.compile(r"\s+")
_QUOTES = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"'})


def normalize_for_scoring(text: str) -> str:
    """Layout-neutral form of one paragraph, used only for scoring.

    - Hard line breaks inside a paragraph (PDF wrapping) become spaces, so a model
      that re-flows lines is not charged with deleting newline tokens.
    - Repeated spaces and trailing spaces collapse to one space.
    - Curly quotes/apostrophes become straight ones, so a model switching ’ to '
      is not counted as a correction.
    The model still receives the original, unnormalized text.
    """
    return _WS.sub(" ", text.translate(_QUOTES)).strip()


def split_paragraphs(text: str) -> list[str]:
    """Paragraphs = blocks separated by a blank line, each normalized for scoring."""
    return [normalize_for_scoring(p) for p in re.split(r"\n\s*\n", text) if p.strip()]


# ----------------------------------------------------------------------------- Ollama


def ollama_chat(model: str, messages: list[dict], options: dict, url: str = config.OLLAMA_URL,
                fmt: dict | str | None = None, logprobs: bool = False) -> dict:
    payload = {"model": model, "messages": messages, "stream": False,
               "options": options, "keep_alive": "30m"}
    if fmt is not None:
        payload["format"] = fmt
    if logprobs:
        payload["logprobs"] = True
        payload["top_logprobs"] = 5
    t0 = time.perf_counter()
    r = requests.post(f"{url}/api/chat", json=payload, timeout=config.REQUEST_TIMEOUT_S)
    r.raise_for_status()
    data = r.json()
    data["wall_s"] = time.perf_counter() - t0
    return data


def ollama_embed(model: str, inputs: list[str], url: str = config.EMBED_URL) -> list[list[float]]:
    r = requests.post(f"{url}/api/embed", json={"model": model, "input": inputs},
                      timeout=config.REQUEST_TIMEOUT_S)
    r.raise_for_status()
    return r.json()["embeddings"]


# ----------------------------------------------------------------------------- RAG + prompts

# The app task class that supplies template, style_context and retrieval_query per prompt.
TASK_BY_PROMPT = {"grammar": GrammarTask}


def retrieve_guides(paths: EvalPaths, item: dict) -> list[str]:
    """Top-k writing-guide chunks for one test item, from the Chroma collection named
    after the style.

    Cached in data/eval/prompts/<prompt>/rag_cache.json under "<style>/<id>|k=<k>", so
    every model gets byte-identical prompts. Delete the cache after changing RAG_TOP_K.
    The query matches the app's (Task.retrieval_query plus the first RAG_QUERY_CHARS of text).
    """
    if not paths.rag:
        return []
    task = TASK_BY_PROMPT[paths.prompt]
    cache = json.loads(paths.rag_cache.read_text(encoding="utf-8")) if paths.rag_cache.exists() else {}
    key = f"{paths.style}/{item['id']}|k={config.RAG_TOP_K}"
    if key in cache:
        return cache[key]

    import chromadb

    client = chromadb.PersistentClient(path=config.CHROMA_PATH)
    collection = client.get_collection(paths.style)
    query = f"{task.retrieval_query}\n\n{item['text'][: config.RAG_QUERY_CHARS]}"
    query_vec = ollama_embed(config.RAG_EMBED_MODEL, [query], url=config.OLLAMA_URL)[0]
    res = collection.query(query_embeddings=[query_vec], n_results=config.RAG_TOP_K)
    guides = list(dict.fromkeys(res["documents"][0]))

    cache[key] = guides
    paths.rag_cache.parent.mkdir(parents=True, exist_ok=True)
    paths.rag_cache.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
    return guides


def build_messages(paths: EvalPaths, item: dict) -> list[dict]:
    """The app's prompt for paths.prompt as one user message, identical for every tested model."""
    task = TASK_BY_PROMPT[paths.prompt]
    guides = "\n\n---\n\n".join(retrieve_guides(paths, item))
    # The guardrail containment_note(canary) the app appends is left out on purpose: its
    # random canary would make every prompt differ and break prompt parity across models.
    prompt = task.template.format(
        style_context=task.style_context[paths.style],
        guidelines=guides, text=item["text"], note="none")
    return [{"role": "user", "content": prompt}]
