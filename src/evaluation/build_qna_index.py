"""Index each Q&A text's userdocs and outline into its own Chroma collections, and list missing material.

    uv run python -m src.evaluation.build_qna_index --style formal --prompt qna

Reads data/eval/<style>/qna/<id>/ for every test text:
    userdocs/*.md     up to config.QNA_MAX_USERDOCS documents the user has besides the text
    outline/*.md      the outline or assignment criterion (one file)
    answers.json      {"<question_id>": ["fact", ...]}, 2-4 facts per question, read by eval_judge

Chunks and ids come from the app's ingestion (src/vector_db: load_documents, split_documents,
add_to_chroma), so the eval indexes documents the way the app does. The embedding function is
pinned to config.RAG_EMBED_MODEL, the model common.retrieve_chunks embeds the questions with.
Collections live in config.QNA_CHROMA_PATH, apart from the app's Chroma, and are named by
common.qna_collection. Reruns add only chunks a collection does not hold yet, so after editing a
document, delete config.QNA_CHROMA_PATH and the Q&A rag_cache.json, then rebuild.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from langchain_ollama import OllamaEmbeddings

from src.vector_db.chunk_documents import split_documents
from src.vector_db.generate_chroma import add_to_chroma
from src.vector_db.load_documents import load_documents

from . import config
from .common import (EvalPaths, UnsupportedPrompt, add_target_args, load_answers, load_questions,
                     load_testset, paths_from_args, qna_collection)

KINDS = ("userdocs", "outline")


def index_text(paths: EvalPaths, text_id: str, embeddings: OllamaEmbeddings) -> list[str]:
    """Ingest one text's userdocs and outline; return the problems found, [] if none."""
    problems = []
    for kind in KINDS:
        folder = paths.qna_dir(text_id) / kind
        docs = load_documents(str(folder)) if folder.is_dir() else []
        if not docs:
            problems.append(f"no {kind} in {folder}")
            continue
        if kind == "userdocs" and len(docs) > config.QNA_MAX_USERDOCS:
            problems.append(f"{len(docs)} userdocs, more than {config.QNA_MAX_USERDOCS}")
        if kind == "outline" and len(docs) > 1:
            problems.append(f"{len(docs)} outline files, expected 1")
        add_to_chroma(split_documents(docs), chroma_path=config.QNA_CHROMA_PATH,
                      collection_name=qna_collection(paths.style, text_id, kind),
                      embedding_function=embeddings)
    return problems


def missing_answers(paths: EvalPaths, text_id: str, question_ids: list[str]) -> list[str]:
    answers = load_answers(paths, text_id)
    problems = [f"no answer for {q}" for q in question_ids if not answers.get(q)]
    problems += [f"{q}: {len(facts)} facts, expected 2-4" for q, facts in answers.items()
                 if q in question_ids and facts and not 2 <= len(facts) <= 4]
    return problems


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_target_args(ap)
    paths = paths_from_args(ap.parse_args())
    if paths.prompt != "qna":
        raise UnsupportedPrompt("build_qna_index builds the Q&A documents only; use --prompt qna.")
    if not paths.rag:
        print("--norag: the no-RAG baseline reads no userdocs or outline, nothing to index.")
        return

    question_ids = [q.id for q in load_questions(paths)]
    embeddings = OllamaEmbeddings(model=config.RAG_EMBED_MODEL, base_url=config.OLLAMA_URL)
    Path(config.QNA_CHROMA_PATH).mkdir(parents=True, exist_ok=True)
    report: dict[str, list[str]] = {}
    for item in load_testset(paths):
        if not paths.qna_dir(item["id"]).is_dir():
            report[item["id"]] = [f"no folder {paths.qna_dir(item['id'])}"]
            continue
        print(f"[{item['id']}]")
        problems = index_text(paths, item["id"], embeddings) + missing_answers(paths, item["id"], question_ids)
        if problems:
            report[item["id"]] = problems

    if report:
        print("\nMissing or unexpected Q&A material:")
        for text_id, problems in report.items():
            print(f"  {text_id}: " + "; ".join(problems))
    else:
        print("\nAll texts have userdocs, an outline and a full answer sheet.")


if __name__ == "__main__":
    main()
