# =============================================================================
# modules/visualize.py
# Module 5: Visualisation
# Generates all charts from the final stereotype and toxicity score CSVs.
# Run after both evaluate_stereotype.py and evaluate_toxicity.py complete.
# =============================================================================

import os, sys, json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.config import OUTPUT_DIR, STEREO_CATEGORIES, TOXICITY_SUBTYPES

MODELS   = ["LLaMA-2-7B", "BLOOM-560M", "Falcon-1B", "Mistral-7B", "GPT-J-6B"]
LANGS    = ["English", "Tamil", "Hindi"]

MODEL_COLORS = ["#1B2A6B", "#0D7C8C", "#F4A418", "#4A5FA5", "#C0392B"]
LANG_COLORS  = ["#1B2A6B", "#0D7C8C", "#F4A418"]

FIG_W, FIG_H = 9.0, 3.8
BAR_W        = 0.15
DPI          = 180

CHART_DIR = os.path.join(OUTPUT_DIR, "charts")
os.makedirs(CHART_DIR, exist_ok=True)


def save_fig(fig, filename):
    path = os.path.join(CHART_DIR, filename)
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


# ── Load scores ───────────────────────────────────────────────────────────────
def load_stereo_scores():
    path = os.path.join(OUTPUT_DIR, "stereotype_scores.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Not found: {path}. Run evaluate_stereotype.py first.")
    return pd.read_csv(path)


def load_toxicity_scores():
    path = os.path.join(OUTPUT_DIR, "toxicity_scores.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Not found: {path}. Run evaluate_toxicity.py first.")
    return pd.read_csv(path)


def get_sbs_value(df, model, language, category=None):
    row = df[(df["model"] == model) & (df["language"] == language)]
    if row.empty:
        return 0.0
    if category is None:
        return float(row["overall_sbs"].values[0])
    col = f"csbs_{category}"
    if col not in row.columns:
        return 0.0
    return float(row[col].values[0])


def get_stbs_value(df, model, language, subtype):
    row = df[(df["model"] == model) & (df["language"] == language)]
    if row.empty:
        return 0.0
    col = f"stbs_{subtype}"
    if col not in row.columns:
        return 0.0
    return float(row[col].values[0])


# ── Chart: SBS by category for one language (grouped by model) ────────────────
def plot_sbs_by_category(stereo_df, language, fig_num, categories):
    short_labels = [c.replace("/", "/\n").replace("-", "\n") for c in categories]
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    x = np.arange(len(categories))
    n = len(MODELS)
    offsets = np.linspace(-(n-1)/2, (n-1)/2, n) * BAR_W

    for mi, (model, color) in enumerate(zip(MODELS, MODEL_COLORS)):
        vals = [get_sbs_value(stereo_df, model, language, cat) for cat in categories]
        ax.bar(x + offsets[mi], vals, BAR_W, label=model,
               color=color, edgecolor="white", linewidth=0.4)

    ax.axhline(50, color="gray", linestyle="--", linewidth=0.9,
               label="Chance (50%)", zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, fontsize=7.5, ha="center")
    ax.set_ylabel("Stereotype Bias Score (%)", fontsize=9)
    ax.set_ylim(0, 105)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%g%%"))
    ax.tick_params(axis="y", labelsize=8)
    ax.legend(fontsize=7.5, ncol=3, loc="upper right",
              framealpha=0.85, edgecolor="#cccccc")
    ax.set_title(
        f"Fig. {fig_num}. Category-level stereotype bias scores (%) — {language}.",
        fontsize=9.5, pad=6
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle=":", linewidth=0.5, color="#dddddd")
    fig.tight_layout()
    return save_fig(fig, f"fig{fig_num}_sbs_{language.lower()}.png")


# ── Chart: TBS by subtype for one language (grouped by model) ─────────────────
def plot_tbs_by_subtype(toxicity_df, language, fig_num, subtypes):
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    x = np.arange(len(subtypes))
    n = len(MODELS)
    offsets = np.linspace(-(n-1)/2, (n-1)/2, n) * BAR_W
    sub_labels = [s.replace("_", " ").title() for s in subtypes]

    for mi, (model, color) in enumerate(zip(MODELS, MODEL_COLORS)):
        vals = [get_stbs_value(toxicity_df, model, language, sub) for sub in subtypes]
        ax.bar(x + offsets[mi], vals, BAR_W, label=model,
               color=color, edgecolor="white", linewidth=0.4)

    ax.set_xticks(x)
    ax.set_xticklabels(sub_labels, fontsize=8.5, rotation=12, ha="right")
    ax.set_ylabel("Toxicity Bias Score", fontsize=9)
    ax.tick_params(axis="y", labelsize=8)
    ax.legend(fontsize=7.5, ncol=3, loc="upper right",
              framealpha=0.85, edgecolor="#cccccc")
    ax.set_title(
        f"Fig. {fig_num}. Subtype-level toxicity bias scores — {language}.",
        fontsize=9.5, pad=6
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle=":", linewidth=0.5, color="#dddddd")
    fig.tight_layout()
    return save_fig(fig, f"fig{fig_num}_tbs_{language.lower()}.png")


# ── Chart: Cross-lingual SBS averages per category ───────────────────────────
def plot_cross_lingual_sbs(stereo_df, fig_num, categories):
    short_labels = [c.replace("/", "/\n").replace("-", "\n") for c in categories]
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    x = np.arange(len(categories))
    offsets_3 = np.array([-BAR_W, 0, BAR_W])

    for li, (lang, color) in enumerate(zip(LANGS, LANG_COLORS)):
        vals = []
        for cat in categories:
            avg = np.mean([get_sbs_value(stereo_df, m, lang, cat) for m in MODELS])
            vals.append(avg)
        ax.bar(x + offsets_3[li], vals, BAR_W * 0.95, label=lang,
               color=color, edgecolor="white", linewidth=0.4)

    ax.axhline(50, color="gray", linestyle="--", linewidth=0.9,
               label="Chance (50%)", zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, fontsize=7.5, ha="center")
    ax.set_ylabel("Avg. SBS (%)", fontsize=9)
    ax.set_ylim(0, 105)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%g%%"))
    ax.tick_params(axis="y", labelsize=8)
    ax.legend(fontsize=8, loc="upper right", framealpha=0.85, edgecolor="#cccccc")
    ax.set_title(
        f"Fig. {fig_num}. Cross-lingual avg. category-level SBS (%) across all five models.",
        fontsize=9.5, pad=6
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle=":", linewidth=0.5, color="#dddddd")
    fig.tight_layout()
    return save_fig(fig, f"fig{fig_num}_sbs_crosslingual.png")


# ── Chart: Cross-lingual TBS averages per subtype ─────────────────────────────
def plot_cross_lingual_tbs(toxicity_df, fig_num, subtypes):
    sub_labels = [s.replace("_", " ").title() for s in subtypes]
    fig, ax = plt.subplots(figsize=(FIG_W, 3.6))
    x = np.arange(len(subtypes))
    offsets_3 = np.array([-BAR_W, 0, BAR_W])

    for li, (lang, color) in enumerate(zip(LANGS, LANG_COLORS)):
        vals = []
        for sub in subtypes:
            avg = np.mean([get_stbs_value(toxicity_df, m, lang, sub) for m in MODELS])
            vals.append(avg)
        ax.bar(x + offsets_3[li], vals, BAR_W * 0.95, label=lang,
               color=color, edgecolor="white", linewidth=0.4)

    ax.set_xticks(x)
    ax.set_xticklabels(sub_labels, fontsize=8.5, rotation=12, ha="right")
    ax.set_ylabel("Avg. Toxicity Bias Score", fontsize=9)
    ax.tick_params(axis="y", labelsize=8)
    ax.legend(fontsize=8, loc="upper right", framealpha=0.85, edgecolor="#cccccc")
    ax.set_title(
        f"Fig. {fig_num}. Cross-lingual avg. subtype-level TBS across all five models.",
        fontsize=9.5, pad=6
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle=":", linewidth=0.5, color="#dddddd")
    fig.tight_layout()
    return save_fig(fig, f"fig{fig_num}_tbs_crosslingual.png")


# ── Overall SBS/TBS (Paper 1 style) ──────────────────────────────────────────
def plot_overall_sbs(stereo_df):
    fig, ax = plt.subplots(figsize=(8, 3.5))
    x = np.arange(len(MODELS))
    w = 0.22
    for li, (lang, color) in enumerate(zip(LANGS, LANG_COLORS)):
        vals = [get_sbs_value(stereo_df, m, lang) for m in MODELS]
        ax.bar(x + (li - 1) * w, vals, w, label=lang, color=color, edgecolor="white")
    ax.axhline(50, color="gray", linestyle="--", linewidth=0.9, label="Chance (50%)")
    ax.set_xticks(x)
    ax.set_xticklabels(MODELS, fontsize=8.5, rotation=10, ha="right")
    ax.set_ylabel("Stereotype Bias Score (%)", fontsize=9)
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%g%%"))
    ax.legend(fontsize=8); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.set_title("Overall SBS per model per language", fontsize=10, pad=6)
    ax.grid(axis="y", linestyle=":", linewidth=0.5, color="#dddddd")
    fig.tight_layout()
    return save_fig(fig, "fig_overall_sbs.png")


def plot_overall_tbs(toxicity_df):
    fig, ax = plt.subplots(figsize=(8, 3.5))
    x = np.arange(len(MODELS))
    w = 0.22
    for li, (lang, color) in enumerate(zip(LANGS, LANG_COLORS)):
        vals = [get_stbs_value(toxicity_df, m, lang, "toxicity") for m in MODELS]
        ax.bar(x + (li - 1) * w, vals, w, label=lang, color=color, edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(MODELS, fontsize=8.5, rotation=10, ha="right")
    ax.set_ylabel("Toxicity Bias Score", fontsize=9)
    ax.legend(fontsize=8); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.set_title("Overall TBS (Toxicity subtype) per model per language", fontsize=10, pad=6)
    ax.grid(axis="y", linestyle=":", linewidth=0.5, color="#dddddd")
    fig.tight_layout()
    return save_fig(fig, "fig_overall_tbs.png")


# ── Master generate function ──────────────────────────────────────────────────
def generate_all_charts():
    """
    Generates all 10 charts (8 for Paper 2 + 2 overall for Paper 1 verification).
    Returns dict of {figure_name: file_path}.
    """
    print("=" * 60)
    print(" Generating all charts")
    print("=" * 60)

    stereo_df  = load_stereo_scores()
    toxicity_df = load_toxicity_scores()

    # Discover categories and subtypes from actual CSV columns
    stereo_cats = [c.replace("csbs_", "") for c in stereo_df.columns if c.startswith("csbs_")]
    tox_subs    = [c.replace("stbs_", "") for c in toxicity_df.columns if c.startswith("stbs_")]

    if not stereo_cats:
        raise ValueError("No category columns found in stereotype_scores.csv. "
                         "Expected columns starting with 'csbs_'.")
    if not tox_subs:
        raise ValueError("No subtype columns found in toxicity_scores.csv. "
                         "Expected columns starting with 'stbs_'.")

    print(f"  Categories detected ({len(stereo_cats)}): {stereo_cats}")
    print(f"  Subtypes detected  ({len(tox_subs)}):    {tox_subs}")
    print()

    paths = {}

    # Figs 1–3: SBS by category per language
    for i, lang in enumerate(LANGS, start=1):
        paths[f"fig{i}_sbs_{lang}"] = plot_sbs_by_category(stereo_df, lang, i, stereo_cats)

    # Figs 4–6: TBS by subtype per language
    for i, lang in enumerate(LANGS, start=4):
        paths[f"fig{i}_tbs_{lang}"] = plot_tbs_by_subtype(toxicity_df, lang, i, tox_subs)

    # Fig 7: Cross-lingual SBS
    paths["fig7_sbs_cross"] = plot_cross_lingual_sbs(stereo_df, 7, stereo_cats)

    # Fig 8: Cross-lingual TBS
    paths["fig8_tbs_cross"] = plot_cross_lingual_tbs(toxicity_df, 8, tox_subs)

    # Overall charts (for Paper 1 / verification)
    paths["overall_sbs"] = plot_overall_sbs(stereo_df)
    paths["overall_tbs"] = plot_overall_tbs(toxicity_df)

    print()
    print(f"  All {len(paths)} charts saved to: {CHART_DIR}")
    return paths


if __name__ == "__main__":
    generate_all_charts()
