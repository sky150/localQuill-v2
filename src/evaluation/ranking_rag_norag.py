"""Compare RAG and no-RAG results of one style and prompt: plots per metric and a composite ranking.

    uv run python -m src.evaluation.ranking_rag_norag --style fiction --prompt grammar

Reads summary_long.csv from results/<prompt>/<style>/ (RAG) and results/<prompt>/<style>-norag/
(no RAG), so run aggregate for both first. A missing condition is skipped.

Writes
    results/<prompt>/<style>/rag_vs_norag.png   one row per metric, left RAG, right no RAG
    results/<prompt>/<variant>/ranking.csv      composite score of every model x temperature

The composite score min-max scales each weighted metric over the rows of ONE condition. A score
compares models within that condition and folder only: never compare it across styles or between
RAG and no RAG (see README.md "Composite ranking").
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write files only, never open a window

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from . import config  # noqa: E402
from .common import EvalPaths, UnsupportedPrompt, eval_paths  # noqa: E402

# Metrics where a LOWER mean is better. Everything else: higher is better.
LOWER_IS_BETTER = {"overcorr_per_100tok", "wall_s", "wall_s_median"}

# Composite score weights. The score divides by the sum of the weights a row has, so they
# need not sum to 1 and the score stays 0-1.
WEIGHTS = {
    "f05_best": 0.30,
    "precision_best": 0.10,
    "tp_rate_consensus": 0.10,
    "overcorr_per_100tok": 0.25,
    "judge_correction": 0.10,
    "judge_preservation": 0.10,
    "gen_tok_per_s": 0.05,
}


GATES = {"cos_doc": 0.95}  # below this = made-up or missing text -> disqualified, no score

# Fixed colour + marker per model (colourblind-checked categorical palette, fixed order).
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
MARKERS = ["o", "s", "^", "D", "v"]


# ----------------------------------------------------------------------------- load


def load_grouped(paths: EvalPaths) -> pd.DataFrame | None:
    """summary_long.csv means as (metric, temperature, model, mean); None if the file is missing."""
    path = paths.result("summary_long.csv")
    if not path.exists():
        print(f"skip {paths.variant}: no {path} (run aggregate first)")
        return None
    df = pd.read_csv(path)
    df["temperature"] = df["setting_id"].str.lstrip("t").astype(float)
    return (
        df.groupby(["metric", "temperature", "model"], as_index=False)["mean"]
        .mean()
        .sort_values(["metric", "model", "temperature"])
    )


def direction(metric: str) -> str:
    return "lower = better" if metric in LOWER_IS_BETTER else "higher = better"


# ----------------------------------------------------------------------------- plot


def plot(conditions: dict[str, pd.DataFrame], title: str, out: Path) -> None:
    """One row per metric, one column per condition; the y-axis is shared per row."""
    all_data = pd.concat(conditions.values())
    models = sorted(all_data["model"].unique())  # across all conditions, so colours match
    metrics = sorted(all_data["metric"].unique())
    style = {m: {"color": PALETTE[i % len(PALETTE)], "marker": MARKERS[i % len(MARKERS)]}
             for i, m in enumerate(models)}

    height = 2.8 * len(metrics)
    fig, axes = plt.subplots(len(metrics), len(conditions), figsize=(5.5 * len(conditions), height),
                             sharey="row", squeeze=False)
    for row, metric in enumerate(metrics):
        for col, (cond_name, grouped) in enumerate(conditions.items()):
            ax = axes[row, col]
            sub = grouped[grouped["metric"] == metric]
            for model in models:
                d = sub[sub["model"] == model]
                if not d.empty:
                    ax.plot(d["temperature"], d["mean"], linewidth=2, markersize=6, **style[model])
            ax.set_title(f"{cond_name} - {metric}  ({direction(metric)})", fontsize=10)
            ax.set_xlabel("temperature")
            if not sub.empty:
                ax.set_xticks(sorted(sub["temperature"].unique()))
            ax.grid(axis="y", alpha=0.3)
            ax.spines[["top", "right"]].set_visible(False)

    handles = [plt.Line2D([], [], linewidth=2, markersize=6, label=m, **style[m]) for m in models]
    fig.suptitle(title, y=1 - 0.1 / height, fontsize=12)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1 - 0.4 / height),
               ncol=len(models), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.8 / height))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------- rank


def metric_ranking(grouped: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Rank models by the mean over temperatures, plus their best single temperature."""
    sub = grouped[grouped["metric"] == metric]
    asc = metric in LOWER_IS_BETTER
    best = (
        sub.sort_values("mean", ascending=asc)
        .groupby("model", as_index=False)
        .first()[["model", "temperature", "mean"]]
        .rename(columns={"temperature": "best_temp", "mean": "best_mean"})
    )
    avg = sub.groupby("model", as_index=False)["mean"].mean().rename(columns={"mean": "avg_mean"})
    out = avg.merge(best, on="model").sort_values("avg_mean", ascending=asc).reset_index(drop=True)
    out.index += 1
    out["best_temp"] = out["best_temp"].map("{:.1f}".format)
    return out


def composite(grouped: pd.DataFrame) -> pd.DataFrame:
    """Every model x temperature with all metric means, gate result, composite score and rank.

    Each weighted metric is min-max scaled to 0-1 over the rows that pass the gates (1 = best row).
    The score is the weighted mean of the scaled metrics a row has, so a row without e.g.
    gen_tok_per_s (LM Studio) still gets a 0-1 score from the rest. Rows that fail a gate keep
    their metric values but get no score.
    """
    wide = grouped.pivot_table(index=["model", "temperature"], columns="metric", values="mean")
    metrics = list(wide.columns)

    gates = {m: v for m, v in GATES.items() if m in wide.columns}
    passed = ((wide[list(gates)] >= pd.Series(gates)).all(axis=1) if gates
              else pd.Series(True, index=wide.index))
    wide["passed_gate"] = passed

    weights = {m: w for m, w in WEIGHTS.items() if m in wide.columns}
    for m in sorted(set(WEIGHTS) - set(weights)):
        print(f"  warning: metric {m!r} missing, left out of the score")

    ok = wide[passed]
    num = pd.Series(0.0, index=ok.index)
    den = pd.Series(0.0, index=ok.index)
    for metric, w in weights.items():
        col = ok[metric]
        rng = col.max() - col.min()
        if pd.notna(rng) and rng > 0:
            norm = (col - col.min()) / rng
            if metric in LOWER_IS_BETTER:
                norm = 1 - norm
        else:  # all rows equal (or one row): the metric does not separate them
            norm = col.where(col.isna(), 1.0)
        num += (w * norm).fillna(0)
        den += norm.notna() * w
    wide["score"] = (num / den.where(den > 0)).reindex(wide.index)

    wide = wide.sort_values("score", ascending=False, na_position="last").reset_index()
    wide.insert(0, "rank", wide["score"].rank(ascending=False, method="min").astype("Int64"))
    best_idx = wide.dropna(subset=["score"]).groupby("model")["score"].idxmax()
    wide["best_for_model"] = wide.index.isin(best_idx)
    cols = ["rank", "model", "temperature", "best_for_model", "passed_gate", "score"]
    return wide[cols + [m for m in metrics if m not in cols]]


# ----------------------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    # Same --style/--prompt as the other scripts. No --norag: this script reads both conditions.
    ap.add_argument("--style", required=True, choices=config.STYLES)
    ap.add_argument("--prompt", required=True, choices=config.PROMPTS)
    args = ap.parse_args()
    # WEIGHTS and the cos_doc gate are grammar metrics (ERRANT and embeddings).
    if args.prompt != "grammar":
        raise UnsupportedPrompt(f"ranking_rag_norag ranks grammar runs only, not {args.prompt!r}.")

    variants = {"RAG": eval_paths(args.style, args.prompt, rag=True),
                "No RAG": eval_paths(args.style, args.prompt, rag=False)}
    loaded = {}
    for name, paths in variants.items():
        grouped = load_grouped(paths)
        if grouped is not None:
            loaded[name] = (paths, grouped)
    if not loaded:
        raise SystemExit("No summary_long.csv for either condition. Run aggregate first.")
    conditions = {name: grouped for name, (_, grouped) in loaded.items()}

    fmt = "{:.4f}".format
    for metric in sorted(pd.concat(conditions.values())["metric"].unique()):
        print(f"\n=== {metric} ({direction(metric)}) ===")
        for cond_name, grouped in conditions.items():
            if (grouped["metric"] == metric).any():
                print(f"--- {cond_name}")
                print(metric_ranking(grouped, metric).to_string(float_format=fmt))

    for cond_name, (paths, grouped) in loaded.items():
        print(f"\n=== Composite - {cond_name} ({paths.variant}), best setting per model ===")
        ranked = composite(grouped)
        out = paths.result("ranking.csv")
        ranked.to_csv(out, index=False)
        shown = ["rank", "model", "temperature", "score"] + [m for m in WEIGHTS if m in ranked.columns]
        print(ranked.loc[ranked["best_for_model"], shown].round(3).to_string(index=False))
        print(f"-> {out}")

    png = variants["RAG"].result("rag_vs_norag.png")
    plot(conditions, f"{args.prompt} / {args.style}: RAG vs. no RAG", png)
    print(f"\n-> {png}")


if __name__ == "__main__":
    main()
