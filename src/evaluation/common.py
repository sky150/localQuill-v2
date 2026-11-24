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

from src.tasks.feedback import FeedbackTask
from src.tasks.grammar import GrammarTask
from src.tasks.qa import QATask
from src.tasks.utility import clip, format_attachments

from . import config

# ----------------------------------------------------------------------------- paths


class UnsupportedPrompt(SystemExit):
    """--prompt names a folder that has no prompt builder or metrics yet."""


class UnsupportedStyle(SystemExit):
    """--style is not evaluated for this prompt (config.STYLES_BY_PROMPT)."""


class MissingParentAnswer(SystemExit):
    """A Q&A follow-up ran before the answer to its parent question existed."""


@dataclass(frozen=True)
class EvalPaths:
    """Every folder and file of one (style, prompt) evaluation.

    data/eval/<style>/                 error_texts/, raw/, opus/, languagetool/, n_errors.json,
                                       agreement.csv, qna/<id>/
    data/eval/prompts/<prompt>/        rag_cache.json, judge_steps.json, questions.json (qna)
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
    def input_texts(self) -> Path:
        """The test texts of this prompt: error_texts/, or raw/ for feedback (config.INPUT_DIR_BY_PROMPT)."""
        return self.style_dir / config.INPUT_DIR_BY_PROMPT[self.prompt]

    def qna_dir(self, text_id: str) -> Path:
        """Q&A material of one text: userdocs/, outline/, answers.json."""
        return self.style_dir / "qna" / text_id

    @property
    def questions_file(self) -> Path:
        return self.prompt_dir / "questions.json"

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
    if style not in config.STYLES_BY_PROMPT[prompt]:
        raise UnsupportedStyle(f"Prompt {prompt!r} is evaluated for styles "
                               f"{config.STYLES_BY_PROMPT[prompt]} only, not {style!r}.")
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
    """One item per <input_texts>/<id>.txt: {"id", "style", "text", "n_errors"}.

    n_errors comes from the optional n_errors.json ({id: count}); None if absent.
    """
    files = sorted(paths.input_texts.glob("*.txt"))
    if not files:
        raise SystemExit(f"No test texts in {paths.input_texts}")
    counts = (json.loads(paths.n_errors_file.read_text(encoding="utf-8"))
              if paths.n_errors_file.exists() else {})
    return [{"id": f.stem, "style": paths.style, "text": f.read_text(encoding="utf-8"),
             "n_errors": counts.get(f.stem)} for f in files]


def testset_sha(paths: EvalPaths) -> str:
    """First 12 hex chars of a hash over all test text names and contents."""
    h = hashlib.sha256()
    for f in sorted(paths.input_texts.glob("*.txt")):
        h.update(f.name.encode() + b"\0" + f.read_bytes() + b"\0")
    return h.hexdigest()[:12]


@dataclass(frozen=True)
class Question:
    """One Q&A question. A follow-up is asked after its parent, with the parent's answer as history."""
    id: str
    question: str
    follow_up_of: str | None = None


class InvalidQuestions(SystemExit):
    """questions.json lists a follow-up before its parent, or names an unknown parent."""


def load_questions(paths: EvalPaths) -> list[Question]:
    """The questions of paths.style from data/eval/prompts/qna/questions.json, in file order."""
    if not paths.questions_file.exists():
        raise SystemExit(f"No questions file {paths.questions_file}")
    raw = json.loads(paths.questions_file.read_text(encoding="utf-8")).get(paths.style, [])
    questions = [Question(q["id"], q["question"], q.get("follow_up_of")) for q in raw]
    seen: set[str] = set()
    for q in questions:
        if q.follow_up_of is not None and q.follow_up_of not in seen:
            raise InvalidQuestions(f"Follow-up {q.id!r} must come after its parent {q.follow_up_of!r} "
                                   f"in {paths.questions_file}.")
        seen.add(q.id)
    if not questions:
        raise SystemExit(f"No {paths.style} questions in {paths.questions_file}")
    return questions


def load_answers(paths: EvalPaths, text_id: str) -> dict[str, list[str]]:
    """The answer sheet of one text, {question_id: [fact, ...]}; {} if it has none yet."""
    path = paths.qna_dir(text_id) / "answers.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def qna_collection(style: str, text_id: str, kind: str) -> str:
    """Chroma collection of one text's userdocs or outline. Hashed, since fiction ids exceed
    Chroma's 63-character name limit."""
    return f"qna_{style}_{hashlib.sha1(text_id.encode()).hexdigest()[:10]}_{kind}"


def load_reference(paths: EvalPaths, name: str, text_id: str) -> str | None:
    path = paths.reference_dir(name) / f"{text_id}.txt"
    return path.read_text(encoding="utf-8") if path.exists() else None


def load_runs(paths: EvalPaths) -> list[dict]:
    runs = read_jsonl(paths.runs_file)
    if not runs:
        raise SystemExit(f"No runs found in {paths.runs_file}. Run run_models first.")
    return runs


def run_key(row: dict) -> tuple:
    """Identifies one generation. question_id is None for grammar and feedback rows."""
    return (row["model"], row["setting_id"], row["run"], row["text_id"], row.get("question_id"))


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
TASK_BY_PROMPT = {"grammar": GrammarTask, "feedback": FeedbackTask, "qna": QATask}


def retrieve_chunks(paths: EvalPaths, key: str, collection: str, query: str, k: int,
                    chroma_path: str = config.CHROMA_PATH, missing_ok: bool = False) -> list[str]:
    """Top-k chunks of a Chroma collection, cached in data/eval/prompts/<prompt>/rag_cache.json
    under `key`, so every model gets byte-identical prompts. Duplicate chunks are dropped.

    missing_ok: a collection that does not exist gives [] and stays out of the cache, so the
    chunks appear once the collection is built.
    """
    cache = json.loads(paths.rag_cache.read_text(encoding="utf-8")) if paths.rag_cache.exists() else {}
    if key in cache:
        return cache[key]

    import chromadb

    client = chromadb.PersistentClient(path=chroma_path)
    if missing_ok:
        # chromadb < 0.6 lists Collection objects, newer versions list names
        names = {c if isinstance(c, str) else c.name for c in client.list_collections()}
        if collection not in names:
            print(f"[rag] no collection {collection} in {chroma_path} ({key}), using none")
            return []
    query_vec = ollama_embed(config.RAG_EMBED_MODEL, [query], url=config.OLLAMA_URL)[0]
    res = client.get_collection(collection).query(query_embeddings=[query_vec], n_results=k)
    chunks = list(dict.fromkeys(res["documents"][0]))

    cache[key] = chunks
    paths.rag_cache.parent.mkdir(parents=True, exist_ok=True)
    paths.rag_cache.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
    return chunks


def retrieve_guides(paths: EvalPaths, item: dict) -> list[str]:
    """Top-k writing-guide chunks for one test item, from the Chroma collection named
    after the style. Cache key "<style>/<id>|k=<k>"; delete the cache after changing RAG_TOP_K.
    The query matches the app's (Task.retrieval_query plus the first RAG_QUERY_CHARS of text).
    """
    if not paths.rag:
        return []
    task = TASK_BY_PROMPT[paths.prompt]
    query = f"{task.retrieval_query}\n\n{item['text'][: config.RAG_QUERY_CHARS]}"
    return retrieve_chunks(paths, f"{paths.style}/{item['id']}|k={config.RAG_TOP_K}",
                           paths.style, query, config.RAG_TOP_K)


def retrieve_qna_docs(paths: EvalPaths, item: dict, question: Question, kind: str) -> str:
    """The text's userdocs or outline chunks for one question, as QATask.build_prompt joins them.
    The query is the question alone, as in the app. '' for the no-RAG baseline."""
    if not paths.rag:
        return ""
    k = QATask.collections[kind]
    key = f"{paths.style}/{item['id']}|{question.id}|{kind}|k={k}"
    chunks = retrieve_chunks(paths, key, qna_collection(paths.style, item["id"], kind),
                             question.question, k, chroma_path=config.QNA_CHROMA_PATH, missing_ok=True)
    return "\n\n---\n\n".join(chunks)


def qna_history(question: Question, parent: Question | None, parent_answer: str | None) -> list[dict]:
    """The conversation before a question: empty, or the parent question and this model's own
    answer to it (same model, setting and run), shaped like the app's chat history."""
    if question.follow_up_of is None:
        return []
    if parent is None or parent_answer is None:
        raise MissingParentAnswer(f"Follow-up {question.id!r} needs the answer to {question.follow_up_of!r} first.")
    return [{"role": "user", "content": parent.question, "task": "qa"},
            {"role": "assistant", "content": parent_answer, "task": "qa"}]


def _grammar_prompt(paths: EvalPaths, item: dict) -> str:
    task = GrammarTask
    guides = "\n\n---\n\n".join(retrieve_guides(paths, item))
    return task.template.format(
        style_context=task.style_context[paths.style],
        guidelines=guides, text=item["text"], note="none")


def _feedback_prompt(paths: EvalPaths, item: dict) -> str:
    # The app also retrieves an outline; the test texts have none, so it reads "none found".
    task = FeedbackTask
    guides = "\n\n---\n\n".join(retrieve_guides(paths, item))
    return task.template.format(
        style_context=task.style_context[paths.style], standards=task.standards[paths.style],
        outline="none found", guidelines=guides or "none found", text=item["text"], note="none")


def _qna_prompt(paths: EvalPaths, item: dict, question: Question, history: list[dict]) -> str:
    task = QATask
    attached = format_attachments([{"name": f"{item['id']}.txt", "content": item["text"]}])
    outline = clip(retrieve_qna_docs(paths, item, question, "outline"), config.QNA_OUTLINE_CHARS)
    userdocs = clip(retrieve_qna_docs(paths, item, question, "userdocs"), config.QNA_USERDOCS_CHARS)
    return task.template.format(
        style_context=task.style_context[paths.style],
        outline=outline or "none found", userdocs=userdocs or "none found",
        current_documents=clip(attached, config.QNA_DOC_CHARS) or "none attached",
        history=task._format_history(history), text=question.question)


def build_messages(paths: EvalPaths, item: dict, question: Question | None = None,
                   history: list[dict] | None = None) -> list[dict]:
    """The app's prompt for paths.prompt as one user message. Grammar and feedback prompts are
    identical for every tested model. A Q&A follow-up carries the model's own earlier answer."""
    # The guardrail containment_note(canary) the app appends is left out on purpose: its
    # random canary would make every prompt differ and break prompt parity across models.
    if paths.prompt == "grammar":
        prompt = _grammar_prompt(paths, item)
    elif paths.prompt == "feedback":
        prompt = _feedback_prompt(paths, item)
    elif paths.prompt == "qna":
        if question is None:
            raise TypeError("Q&A prompts need a question")
        prompt = _qna_prompt(paths, item, question, history or [])
    else:
        raise UnsupportedPrompt(f"No prompt builder for {paths.prompt!r}.")
    return [{"role": "user", "content": prompt}]
