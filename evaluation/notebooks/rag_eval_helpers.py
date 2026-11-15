"""Shared loaders and chart style for 8_rag_evaluation.ipynb and 9_embedding_model_decision.ipynb.

The notebooks only read files in evaluation/reports/ (and git history for the old prompts);
they never re-run an eval.
"""

import json
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import ListedColormap

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "pyproject.toml").exists())
sys.path.insert(0, str(ROOT / "tests" / "eval" / "rag"))
from run_rag_eval import APP_TOP_K, K_MAX, metrics  # noqa: E402  same metric code as the eval

REPORTS = ROOT / "evaluation" / "reports"
RAG = REPORTS / "RAG"
TESTS = ROOT / "tests" / "eval" / "rag"
BEFORE_COMMIT = "1f8295c"  # main before this work: the "before" versions of prompts and config

# Validated categorical colours in fixed order (blue, orange, aqua, yellow, magenta) and recessive ink/grid.
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
SERIES = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA]
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SCORE_CMAP = ListedColormap(["#cde2fb", "#6da7ec", "#1c5cab"])  # 0 / 1 / 2, one hue light->dark
RAMP_CMAP = ListedColormap(["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95"])  # 0..1

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": GRID,
    "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2, "text.color": INK,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
    "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold", "legend.frameon": False,
})
pd.set_option("display.max_colwidth", 120)

# Search queries for the style guides, named after where they come from in the code
# (run_rag_eval.py stores them under the short keys).
QUERY_NAMES = {
    "task": "keywords + text (app before)",  # retrieval_query + text[:2000]
    "draft": "text only",                    # text[:2000], the first 2,000 characters of the user's text
    "keywords": "keywords only",             # retrieval_query in feedback.py / grammar.py
    "llm": "query written by the model",     # the old free-text STYLE_QUERY_PROMPT (run 5)
    "issue": "test set answer (best case)",  # the `issue` field of testset_style.json; the app can't know it
    "label": "label picked by the model",    # style_query: 1 item from problem_labels (app now)
}

GRAMMAR = {"grammar/formal", "grammar/fiction"}
FEEDBACK = {"feedback/formal", "feedback/fiction"}


# ---------- data ----------

def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def read_concatenated_json(path: Path) -> list[dict]:
    """The old DISPRO1 file has objects written back to back, not one per line."""
    text, decoder, i, rows = path.read_text(), json.JSONDecoder(), 0, []
    while i < len(text):
        if text[i].isspace():
            i += 1
            continue
        obj, i = decoder.raw_decode(text, i)
        rows.append(obj)
    return rows


RUNS = read_jsonl(RAG / "results.jsonl")
QA_SET = json.loads((TESTS / "testset_qa.json").read_text())
STYLE_SET = json.loads((TESTS / "testset_style.json").read_text())


def run(model="nomic-embed-text", chunk=1000, overlap=400, prefixes=False) -> dict:
    return next(r for r in RUNS if (r["model"], r["chunk_size"], r["overlap"], r["prefixes"])
                == (model, chunk, overlap, prefixes))


def cases_of(r: dict, set_name: str, groups: set, query_style: str) -> list[dict]:
    return [c for c in r["cases"] if c["set"] == set_name and c["group"] in groups
            and c["query_style"] == query_style]


def score(r, set_name, groups, query_style, k, metric="hit") -> float:
    cases = cases_of(r, set_name, groups, query_style)
    return metrics(cases, k)[metric] if cases else float("nan")


def hand_grades(path: Path, variants: list[str]) -> pd.DataFrame:
    data = json.loads(path.read_text())
    data.pop("_note")
    return pd.DataFrame(data, index=variants).T


def grades_long(path: Path, run: str) -> pd.DataFrame:
    """Hand grades stored as {case: {variant: [repeat scores]}} as one row per answer."""
    data = json.loads(path.read_text())
    data.pop("_note")
    return pd.DataFrame([{"run": run, "id": case, "variant": v, "repeat": i, "score": s}
                         for case, by_variant in data.items() for v, scores in by_variant.items()
                         for i, s in enumerate(scores)])


def git_show(path: str, commit: str = BEFORE_COMMIT) -> str:
    return subprocess.run(["git", "show", f"{commit}:{path}"], cwd=ROOT, capture_output=True,
                          text=True, check=True).stdout


# ---------- charts ----------

def grouped_bars(ax, df: pd.DataFrame, colors, ylabel: str, ylim=(0, 1)):
    """Thin grouped bars with a gap between neighbours; a 0 gets a visible label."""
    n, width = len(df.columns), 0.8 / len(df.columns)
    for i, (col, color) in enumerate(zip(df.columns, colors)):
        xs = [x + (i - (n - 1) / 2) * width for x in range(len(df))]
        ax.bar(xs, df[col], width * 0.9, color=color, label=col)
        for x, v in zip(xs, df[col]):
            if v == 0:  # an empty slot would read as missing data
                ax.annotate("0", (x, 0), xytext=(0, 3), textcoords="offset points", ha="center", color=INK_2)
    ax.set_xticks(range(len(df)), df.index)
    ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel)


def legend_on_top(ax, ncol=2):
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=ncol)


def score_grid(ax, grid: pd.DataFrame, vmax=2, fmt="{:.0f}", cmap=SCORE_CMAP, vmin=0):
    """Rows x columns heatmap, darker = better, value printed in every cell."""
    ax.imshow(grid.values.astype(float), cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            v = grid.iat[i, j]
            ax.text(j, i, fmt.format(v), ha="center", va="center", color="white" if v > vmin + (vmax - vmin) * 0.6 else INK)
    ax.set_xticks(range(grid.shape[1]), grid.columns)
    ax.set_yticks(range(grid.shape[0]), grid.index)
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False, length=0)
    ax.grid(False)
