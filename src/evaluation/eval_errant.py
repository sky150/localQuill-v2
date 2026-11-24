"""ERRANT test: model edits vs. LanguageTool and Opus reference edits.

    uv run python -m spacy download en_core_web_sm   # once
    uv run python -m src.evaluation.eval_errant --style formal --prompt grammar

All edits are extracted against the ORIGINAL text, so an edit is the key
(paragraph, o_start, o_end, correction). Two edits match only if they change
the same original tokens into the same string.

Per text and run it writes (data/eval/results/<prompt>/<style>/errant.csv):
  tp_lt, tp_opus   model edits that match that reference
  tp_any           model edits that match at least one reference
  tp/n_err_spans   errors = unique original spans any reference edited;
                   tp_spans = spans where the model matched a reference edit
  tp_cons          model edits that match an edit BOTH references made
  n_cons           edits both references agree on
  tp/fp/fn_best    errant_compare style: per paragraph, the reference giving the higher F0.5
  fp_union         model edits that NO reference made (over-correction)
  n_errors         known injected error count from the test set (if given)

Texts are aligned by paragraph. If any text has a different paragraph count
than the original, that item falls back to whole-text alignment (mode column).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from functools import lru_cache

import errant
import pandas as pd

from . import config
from .common import (EvalPaths, add_target_args, load_reference, load_runs, load_testset,
                     paths_from_args, split_paragraphs)

annotator = errant.load("en")


@lru_cache(maxsize=None)
def parse(text: str):
    return annotator.parse(text, tokenise=True)


@lru_cache(maxsize=None)
def edits(orig: str, cor: str) -> dict[tuple, str]:
    """{(o_start, o_end, c_str): error_type} for orig -> cor."""
    if orig == cor:
        return {}
    return {(e.o_start, e.o_end, e.c_str): e.type for e in annotator.annotate(parse(orig), parse(cor))}


def segment(orig: str, others: list[str]) -> tuple[str, list[str], list[list[str]]]:
    o = split_paragraphs(orig)
    splits = [split_paragraphs(t) for t in others]
    if all(len(s) == len(o) for s in splits):
        return "paragraph", o, splits
    return "whole", [" ".join(o)], [[" ".join(s)] for s in splits]


def f05(tp: int, fp: int, fn: int) -> float:
    p = tp / (tp + fp) if tp + fp else 1.0
    r = tp / (tp + fn) if tp + fn else 1.0
    return 1.25 * p * r / (0.25 * p + r) if p + r else 0.0


def score_item(item: dict, hyp: str, refs: dict[str, str]) -> dict:
    names = list(refs)
    mode, o_segs, other_segs = segment(item["text"], [hyp] + [refs[n] for n in names])
    hyp_segs, ref_segs = other_segs[0], dict(zip(names, other_segs[1:]))

    c = Counter()
    tp_types = Counter()
    for i, o in enumerate(o_segs):
        h = edits(o, hyp_segs[i])
        r = {n: edits(o, ref_segs[n][i]) for n in names}
        hk = set(h)
        union = set().union(*(set(v) for v in r.values()))
        c["n_hyp"] += len(hk)
        c["fp_union"] += len(hk - union)
        c["tp_any"] += len(hk & union)
        # error = unique original span that any reference edited (two refs fixing the
        # same span differently count once); fixed = model matched a reference there
        c["n_err_spans"] += len({k[:2] for k in union})
        c["tp_spans"] += len({k[:2] for k in hk & union})
        for n in names:
            c[f"tp_{n}"] += len(hk & set(r[n]))
            c[f"n_{n}"] += len(r[n])
        if len(names) == 2:
            cons = set(r[names[0]]) & set(r[names[1]])
            c["n_cons"] += len(cons)
            c["tp_cons"] += len(hk & cons)
        # best reference for this paragraph
        best = max(names, key=lambda n: (f05(len(hk & set(r[n])), len(hk - set(r[n])), len(set(r[n]) - hk)),
                                         len(hk & set(r[n]))))
        rb = set(r[best])
        c["tp_best"] += len(hk & rb)
        c["fp_best"] += len(hk - rb)
        c["fn_best"] += len(rb - hk)
        tp_types.update(h[k] for k in hk & union)

    row = {"mode": mode, "orig_tokens": len(parse(" ".join(o_segs))), **c,
           "n_errors": item.get("n_errors"), "tp_types": json.dumps(dict(tp_types))}
    if len(names) < 2:
        row["n_cons"] = row["tp_cons"] = None
    return row


def reference_agreement(paths: EvalPaths, items: list[dict]) -> None:
    rows = []
    for item in items:
        refs = {n: load_reference(paths, n, item["id"]) for n in config.REFERENCES}
        refs = {k: v for k, v in refs.items() if v is not None}
        if len(refs) < 2:
            continue
        row = score_item(item, refs[config.REFERENCES[0]], refs)   # LT as "hypothesis" vs both
        rows.append({"text_id": item["id"], "n_lt": row[f"n_{config.REFERENCES[0]}"],
                     "n_opus": row[f"n_{config.REFERENCES[1]}"], "n_cons": row["n_cons"],
                     "n_errors": item.get("n_errors")})
    if rows:
        out = paths.agreement_file
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"Reference agreement -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    add_target_args(ap)
    paths = paths_from_args(ap.parse_args())

    items = {i["id"]: i for i in load_testset(paths)}
    reference_agreement(paths, list(items.values()))

    rows = []
    runs = load_runs(paths)
    for k, run in enumerate(runs, 1):
        item = items[run["text_id"]]
        refs = {n: load_reference(paths, n, item["id"]) for n in config.REFERENCES}
        refs = {n: t for n, t in refs.items() if t is not None}
        if not refs:
            print(f"skip {item['id']}: no reference files")
            continue
        rows.append({"model": run["model"], "setting_id": run["setting_id"], "run": run["run"],
                     "text_id": item["id"], "style": item["style"],
                     **score_item(item, run["output"], refs)})
        if k % 25 == 0:
            print(f"[{k}/{len(runs)}]")

    out = paths.result("errant.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"{len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
