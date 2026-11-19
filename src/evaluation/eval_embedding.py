"""Embedding test: semantic similarity between original and returned text.

    uv run python -m src.evaluation.eval_embedding --style formal --prompt grammar

A low score means the model changed content, not just grammar.
Per text and run (embedding.csv in the results dir):
  cos_doc    cosine of mean-pooled paragraph vectors (whole text)
  cos_mean   mean paragraph-pair cosine   (only if paragraph counts match)
  cos_min    worst paragraph-pair cosine  (catches one rewritten paragraph)

Calibration (embedding_refs.csv): the same scores for original vs.
each reference. A model close to the reference scores changed about as much
as a good correction needs to.
Paragraphs are embedded separately because long texts exceed the embedding
model's context window.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from . import config
from .common import (add_target_args, load_reference, load_runs, load_testset, ollama_embed,
                     paths_from_args, split_paragraphs)

_cache: dict[str, np.ndarray] = {}


def embed(texts: list[str]) -> np.ndarray:
    missing = [t for t in dict.fromkeys(texts) if t not in _cache]
    for start in range(0, len(missing), 32):
        batch = missing[start:start + 32]
        vecs = ollama_embed(config.EMBED_MODEL, [config.EMBED_PREFIX + t for t in batch])
        for t, v in zip(batch, vecs):
            v = np.asarray(v, dtype=np.float32)
            _cache[t] = v / (np.linalg.norm(v) or 1.0)
    return np.stack([_cache[t] for t in texts])


def similarity(orig: str, other: str) -> dict:
    po, pc = split_paragraphs(orig), split_paragraphs(other)
    if not po or not pc:
        return {"cos_doc": 0.0, "cos_mean": None, "cos_min": None, "para_match": False}
    eo, ec = embed(po), embed(pc)
    do, dc = eo.mean(0), ec.mean(0)
    row = {"cos_doc": float(do @ dc / (np.linalg.norm(do) * np.linalg.norm(dc))),
           "cos_mean": None, "cos_min": None, "para_match": len(po) == len(pc)}
    if row["para_match"]:
        pair = (eo * ec).sum(1)
        row["cos_mean"], row["cos_min"] = float(pair.mean()), float(pair.min())
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    add_target_args(ap)
    paths = paths_from_args(ap.parse_args())

    items = {i["id"]: i for i in load_testset(paths)}
    paths.results_dir.mkdir(parents=True, exist_ok=True)

    ref_rows = []
    for item in items.values():
        for name in config.REFERENCES:
            ref = load_reference(paths, name, item["id"])
            if ref is not None:
                ref_rows.append({"text_id": item["id"], "style": item["style"],
                                 "reference": name, **similarity(item["text"], ref)})
    pd.DataFrame(ref_rows).to_csv(paths.result("embedding_refs.csv"), index=False)

    rows = []
    runs = load_runs(paths)
    for k, run in enumerate(runs, 1):
        item = items[run["text_id"]]
        rows.append({"model": run["model"], "setting_id": run["setting_id"], "run": run["run"],
                     "text_id": item["id"], "style": item["style"],
                     **similarity(item["text"], run["output"])})
        if k % 50 == 0:
            print(f"[{k}/{len(runs)}]")

    out = paths.result("embedding.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"{len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
