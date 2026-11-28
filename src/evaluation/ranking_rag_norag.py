"""Compare RAG and no-RAG results of one style and prompt: plots per metric and a composite ranking.
Ranks every prompt in WEIGHTS_BY_PROMPT.

    uv run python -m src.evaluation.ranking_rag_norag --style fiction --prompt grammar

Reads summary_long.csv from results/<prompt>/<style>/ (RAG) and results/<prompt>/<style>-norag/
(no RAG), so run aggregate for both first. A missing condition is skipped.

Writes
    results/<prompt>/<style>/rag_vs_norag.png   one row per metric, left RAG, right no RAG
    results/<prompt>/<variant>/ranking.csv      composite score of every model x setting

The composite score min-max scales each weighted metric over the rows of ONE condition. A score
compares models within that condition and folder only: never compare it across styles or between
RAG and no RAG (see README.md "Composite ranking").

A setting_id like "t0.2" is a temperature. Others (OpenAI "effort_low") have no temperature: their
temperature stays NaN, the ranking keeps them per setting_id, and the plots draw them as horizontal
lines across the temperature axis.
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
LOWER_IS_BETTER = {"overcorr_per_100tok", "wall_s", "wall_s_median", "capped_rate"}

# Composite score weights and gates per prompt. The score divides by the sum of the weights a
# row has, so they need not sum to 1 and the score stays 0-1.
# wall_s stays out of qna and feedback: gpt-5-nano's wall_s includes the network round trip, so
# it does not compare with local timings.
WEIGHTS_BY_PROMPT = {
    "grammar": {
        "f05_best": 0.30,
        "precision_best": 0.10,
        "tp_rate_consensus": 0.10,
        "overcorr_per_100tok": 0.25,
        "judge_correction": 0.10,
        "judge_preservation": 0.10,
        "gen_tok_per_s": 0.05,
    },
    # Judge scores 1-5. gpt-5-nano has no gen_tok_per_s, so its score comes from the rest.
    "qna": {
        "judge_correctness": 0.45,
        "judge_groundedness": 0.45,
        "gen_tok_per_s": 0.1,
    },
    # DISPRO1's deepeval metrics, 0-1.
    "feedback": {
        "judge_clarity": 0.45,
        "judge_style": 0.45,
        "gen_tok_per_s": 0.1,
    },
}

# A gate passes at value >= threshold, or <= threshold for LOWER_IS_BETTER metrics.
# cos_doc below 0.95 = made-up or missing text -> disqualified, no score.
# capped_rate 0.1 = rate of cutoff ansers -> Disqualified if more than 10% since they do not respond within required context
GATES_BY_PROMPT = {
    "grammar": {"cos_doc": 0.95, "capped_rate": 0.1},
    "qna": {"capped_rate": 0.1},
    "feedback": {},
}

# Fixed colour + marker per model (colourblind-checked categorical palette, fixed order).
# Six entries because grammar already compares five local models plus gpt-5-nano.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#7b4fa3"]
MARKERS = ["o", "s", "^", "D", "v", "P"]
# One line style per non-temperature setting of a model, so two efforts of one model stay apart.
LINESTYLES = ["--", ":", "-."]


# ----------------------------------------------------------------------------- load


def load_grouped(paths: EvalPaths) -> pd.DataFrame | None:
    """summary_long.csv means as (metric, setting_id, temperature, model, mean); None if the file is missing.

    temperature is NaN for settings that are not a temperature ("effort_low").
    """
    path = paths.result("summary_long.csv")
    if not path.exists():
        print(f"skip {paths.variant}: no {path} (run aggregate first)")
        return None
    df = pd.read_csv(path)
    # Group on setting_id: groupby drops NaN keys, which temperature is for effort settings.
    out = df.groupby(["metric", "setting_id", "model"], as_index=False)["mean"].mean()
    out["temperature"] = pd.to_numeric(out["setting_id"].str.removeprefix("t"), errors="coerce")
    return (
        out[["metric", "setting_id", "temperature", "model", "mean"]]
        .sort_values(["metric", "model", "temperature", "setting_id"])
        .reset_index(drop=True)
    )


def direction(metric: str) -> str:
    return "lower = better" if metric in LOWER_IS_BETTER else "higher = better"


# ----------------------------------------------------------------------------- plot


def plot(conditions: dict[str, pd.DataFrame], title: str, out: Path) -> None:
    """One row per metric, one column per condition; the y-axis is shared per row."""
    all_data = pd.concat(conditions.values())
    # Across all conditions, so colours match. GPT models sort last so adding one keeps the
    # colours of the local models.
    models = sorted(all_data["model"].unique(), key=lambda m: (m.startswith("gpt-"), m))
    metrics = sorted(all_data["metric"].unique())
    style = {m: {"color": PALETTE[i % len(PALETTE)], "marker": MARKERS[i % len(MARKERS)]}
             for i, m in enumerate(models)}

    # Settings without a temperature (effort_low), per model, in a stable order.
    flat_settings = {m: sorted(all_data.loc[(all_data["model"] == m) & all_data["temperature"].isna(),
                                            "setting_id"].unique()) for m in models}
    flat_style = {(m, sid): LINESTYLES[i % len(LINESTYLES)]
                  for m, sids in flat_settings.items() for i, sid in enumerate(sids)}

    height = 2.8 * len(metrics)
    fig, axes = plt.subplots(len(metrics), len(conditions), figsize=(5.5 * len(conditions), height),
                             sharey="row", squeeze=False)
    for row, metric in enumerate(metrics):
        for col, (cond_name, grouped) in enumerate(conditions.items()):
            ax = axes[row, col]
            sub = grouped[grouped["metric"] == metric]
            for model in models:
                d = sub[sub["model"] == model]
                line = d[d["temperature"].notna()]
                if not line.empty:
                    ax.plot(line["temperature"], line["mean"], linewidth=2, markersize=6, **style[model])
                for setting_id in flat_settings[model]:
                    flat = d[d["setting_id"] == setting_id]
                    if not flat.empty:
                        ax.axhline(flat["mean"].iloc[0], linewidth=2, alpha=0.9,
                                   color=style[model]["color"], linestyle=flat_style[(model, setting_id)])
            ax.set_title(f"{cond_name} - {metric}  ({direction(metric)})", fontsize=10)
            ax.set_xlabel("temperature")
            temps = sub["temperature"].dropna().unique()
            if len(temps):
                ax.set_xticks(sorted(temps))
            ax.grid(axis="y", alpha=0.3)
            ax.spines[["top", "right"]].set_visible(False)

    handles = [plt.Line2D([], [], linewidth=2, markersize=6, label=m, **style[m])
               for m in models if (all_data["model"] == m).any() and all_data.loc[
                   all_data["model"] == m, "temperature"].notna().any()]
    handles += [plt.Line2D([], [], linewidth=2, color=style[m]["color"], linestyle=flat_style[(m, sid)],
                           label=f"{m} {sid}")
                for m in models for sid in flat_settings[m]]
    fig.suptitle(title, y=1 - 0.1 / height, fontsize=12)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1 - 0.4 / height),
               ncol=len(models), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.8 / height))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------- rank


def metric_ranking(grouped: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Rank models by the mean over settings, plus their best single setting.

    best_temp is the temperature, or the setting_id when the setting has none ("effort_low").
    """
    sub = grouped[grouped["metric"] == metric]
    asc = metric in LOWER_IS_BETTER
    best = (
        sub.sort_values("mean", ascending=asc)
        .groupby("model", as_index=False)
        .first()[["model", "setting_id", "temperature", "mean"]]
        .rename(columns={"mean": "best_mean"})
    )
    avg = sub.groupby("model", as_index=False)["mean"].mean().rename(columns={"mean": "avg_mean"})
    out = avg.merge(best, on="model").sort_values("avg_mean", ascending=asc).reset_index(drop=True)
    out.index += 1
    out["best_temp"] = [f"{t:.1f}" if pd.notna(t) else sid
                        for t, sid in zip(out["temperature"], out["setting_id"])]
    return out[["model", "avg_mean", "best_temp", "best_mean"]]


def composite(grouped: pd.DataFrame, prompt: str = "grammar") -> pd.DataFrame:
    """Every model x setting with all metric means, gate result, composite score and rank.

    Each weighted metric is min-max scaled to 0-1 over the rows that pass the gates (1 = best row).
    The score is the weighted mean of the scaled metrics a row has, so a row without e.g.
    gen_tok_per_s (LM Studio) still gets a 0-1 score from the rest. Rows that fail a gate keep
    their metric values but get no score.
    """
    # Index on setting_id: pivot_table drops NaN index keys, which temperature is for effort settings.
    wide = grouped.pivot_table(index=["model", "setting_id"], columns="metric", values="mean")
    metrics = list(wide.columns)

    weights_all = WEIGHTS_BY_PROMPT[prompt]
    gates = {m: v for m, v in GATES_BY_PROMPT.get(prompt, {}).items() if m in wide.columns}
    passed = pd.Series(True, index=wide.index)
    for m, threshold in gates.items():
        passed &= (wide[m] <= threshold) if m in LOWER_IS_BETTER else (wide[m] >= threshold)
    wide["passed_gate"] = passed
    if passed.sum() < 2:
        print(f"  warning: {int(passed.sum())} row(s) pass the gate, so min-max scaling gives "
              "every row score 1 and the rank says nothing")

    weights = {m: w for m, w in weights_all.items() if m in wide.columns}
    for m in sorted(set(weights_all) - set(weights)):
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
    temps = grouped.drop_duplicates("setting_id").set_index("setting_id")["temperature"]
    wide.insert(2, "temperature", wide["setting_id"].map(temps))
    wide.insert(0, "rank", wide["score"].rank(ascending=False, method="min").astype("Int64"))
    best_idx = wide.dropna(subset=["score"]).groupby("model")["score"].idxmax()
    wide["best_for_model"] = wide.index.isin(best_idx)
    cols = ["rank", "model", "setting_id", "temperature", "best_for_model", "passed_gate", "score"]
    return wide[cols + [m for m in metrics if m not in cols]]


# ----------------------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    # Same --style/--prompt as the other scripts. No --norag: this script reads both conditions.
    ap.add_argument("--style", required=True, choices=config.STYLES)
    ap.add_argument("--prompt", required=True, choices=config.PROMPTS)
    args = ap.parse_args()
    if args.prompt not in WEIGHTS_BY_PROMPT:
        raise UnsupportedPrompt(f"No weights for {args.prompt!r} in WEIGHTS_BY_PROMPT; "
                                f"ranked prompts: {sorted(WEIGHTS_BY_PROMPT)}.")

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
        ranked = composite(grouped, prompt=args.prompt)
        out = paths.result("ranking.csv")
        ranked.to_csv(out, index=False)
        shown = ["rank", "model", "setting_id", "temperature", "score"] + [m for m in WEIGHTS_BY_PROMPT[args.prompt] if m in ranked.columns]
        print(ranked.loc[ranked["best_for_model"], shown].round(3).to_string(index=False))
        print(f"-> {out}")

    png = variants["RAG"].result("rag_vs_norag.png")
    plot(conditions, f"{args.prompt} / {args.style}: RAG vs. no RAG", png)
    print(f"\n-> {png}")


if __name__ == "__main__":
    main()
