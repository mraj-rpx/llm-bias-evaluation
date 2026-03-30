#!/usr/bin/env python
# coding: utf-8

# # LLM Bias Evaluation Pipeline — Category-Level Analysis
# **Author:** Mohanraj Ramanujam | PA2312049010014 | M.Tech AI, SRM University
# **Guide:** Dr. S. Godfrey Winster
#
# ## Pipeline Overview
# ```
# [1] Validate CSVs          → Check column names and data quality
# [2] Translate Datasets     → English → Tamil (IndicTrans2) + Hindi (NLLB)
# [3] Stereotype Evaluation  → Log-likelihood scoring per category per model per language
# [4] Toxicity Evaluation    → Text generation + Detoxify subtype scoring
# [5] Visualise Results      → Generate all 8+ charts
# [6] Summary Report         → Print all scores in table format
# ```
#
# **Checkpointing:** Each step saves progress. If the run crashes, re-run from the
# same cell — completed work will be detected and skipped automatically.
#
# **Estimated runtime on RTX 4090:** ~6–8 hours for all 5 models × 3 languages.

# ─────────────────────────────────────────────────────────────────────────────
# CELL 1: Imports and path setup
# ─────────────────────────────────────────────────────────────────────────────
import os, sys, json
import pandas as pd

# Add pipeline root to path
PIPELINE_ROOT = os.path.dirname(os.path.abspath("__file__"))
sys.path.insert(0, PIPELINE_ROOT)

from modules.config import (
    OUTPUT_DIR, CHECKPOINT_DIR, MODEL_CONFIGS, LANGUAGES,
    STEREOTYPE_CSV, TOXICITY_CSV,
)
from modules.validate_data   import run_validation
from modules.translate       import translate_datasets
from modules.evaluate_stereotype import run_stereotype_evaluation
from modules.evaluate_toxicity   import run_toxicity_evaluation
from modules.visualize           import generate_all_charts

print("Pipeline root:", PIPELINE_ROOT)
print("Output dir:   ", OUTPUT_DIR)
print("Checkpoint dir:", CHECKPOINT_DIR)
print()
print("Models to evaluate:")
for name in MODEL_CONFIGS:
    cfg = MODEL_CONFIGS[name]
    print(f"  {name:15s}  {cfg['repo']}")

# ─────────────────────────────────────────────────────────────────────────────
# CELL 2: STEP 1 — Validate input CSVs
# ─────────────────────────────────────────────────────────────────────────────
# Place your files in:
#   pipeline/data/stereo_type_dataset.csv
#   pipeline/data/toxicity.csv
#
# Required columns:
#   stereo_type_dataset.csv → sent_more, sent_less, bias_type
#   toxicity.csv            → prompt
#
# If your columns have different names, update config.py:
#   STEREO_COL_MORE, STEREO_COL_LESS, STEREO_COL_CAT, TOXICITY_COL_PROMPT

stereo_ok, tox_ok, stereo_df, tox_df = run_validation()

assert stereo_ok, "Fix stereotype CSV errors above before continuing."
assert tox_ok,    "Fix toxicity CSV errors above before continuing."

print("\n✓ Both CSVs validated. Proceeding to translation.")

# ─────────────────────────────────────────────────────────────────────────────
# CELL 3: STEP 2 — Translate datasets to Tamil and Hindi
# ─────────────────────────────────────────────────────────────────────────────
# Outputs (saved to outputs/):
#   stereo_English.csv, stereo_Tamil.csv, stereo_Hindi.csv
#   toxicity_English.csv, toxicity_Tamil.csv, toxicity_Hindi.csv
#
# This step is SKIPPED automatically if output files already exist.
# Estimated time: ~45–90 minutes depending on dataset size.

translate_datasets(stereo_df, tox_df)
print("\n✓ Translation complete.")

# ─────────────────────────────────────────────────────────────────────────────
# CELL 4: STEP 3 — Stereotype evaluation (all models, all languages)
# ─────────────────────────────────────────────────────────────────────────────
# For each model × language:
#   - Computes log P(sent_more) and log P(sent_less) for every pair
#   - Labels pair as stereotypical if log P(sent_more) > log P(sent_less)
#   - Computes CSBS per category and overall SBS
#
# Checkpoints saved per model × language to checkpoints/stereo_*.csv
# Final outputs:
#   outputs/stereo_raw_predictions.csv
#   outputs/stereotype_scores.json
#   outputs/stereotype_scores.csv
#
# To run only specific models, change model_names list, e.g.:
#   model_names = ["LLaMA-2-7B", "Mistral-7B"]
# To run only specific languages:
#   languages = ["English"]

model_names = list(MODEL_CONFIGS.keys())   # all 5 models
languages   = LANGUAGES                    # English, Tamil, Hindi

stereo_scores, stereo_scores_df = run_stereotype_evaluation(
    model_names=model_names,
    languages=languages,
)

print("\n✓ Stereotype evaluation complete.")
print("\nOverall SBS summary:")
print(stereo_scores_df[["model", "language", "overall_sbs"]].to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────
# CELL 5: STEP 4 — Toxicity evaluation (all models, all languages)
# ─────────────────────────────────────────────────────────────────────────────
# For each model × language:
#   - Generates 50-token greedy continuation per prompt
#   - Scores continuation with Detoxify (6 subtypes)
#   - Computes STBS (mean per subtype)
#
# Checkpoints saved to:
#   checkpoints/toxicity_cont_<model>_<lang>.csv   (continuations)
#   checkpoints/toxicity_scored_<model>_<lang>.csv  (with subtype scores)
# Final outputs:
#   outputs/toxicity_raw_scored.csv
#   outputs/toxicity_scores.json
#   outputs/toxicity_scores.csv

tox_scores, tox_scores_df = run_toxicity_evaluation(
    model_names=model_names,
    languages=languages,
)

print("\n✓ Toxicity evaluation complete.")
print("\nTBS summary (Toxicity subtype):")
tbs_summary = tox_scores_df[["model", "language", "stbs_toxicity"]].copy()
print(tbs_summary.to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────
# CELL 6: STEP 5 — Generate all charts
# ─────────────────────────────────────────────────────────────────────────────
# Reads from outputs/stereotype_scores.csv and outputs/toxicity_scores.csv
# Saves all charts to outputs/charts/

chart_paths = generate_all_charts()

print("\n✓ All charts generated:")
for name, path in chart_paths.items():
    print(f"  {name}: {os.path.basename(path)}")

# ─────────────────────────────────────────────────────────────────────────────
# CELL 7: STEP 6 — Print full summary report
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print(" FULL RESULTS SUMMARY")
print("="*70)

stereo_df_out  = pd.read_csv(os.path.join(OUTPUT_DIR, "stereotype_scores.csv"))
toxicity_df_out = pd.read_csv(os.path.join(OUTPUT_DIR, "toxicity_scores.csv"))

# ── Stereotype: overall SBS ───────────────────────────────────────────────────
print("\n── OVERALL STEREOTYPE BIAS SCORE (SBS %) ──")
pivot_sbs = stereo_df_out.pivot(index="model", columns="language", values="overall_sbs")
print(pivot_sbs.to_string())

# ── Stereotype: per-category averages across models ───────────────────────────
cat_cols = [c for c in stereo_df_out.columns if c.startswith("csbs_")]
print(f"\n── CATEGORY-LEVEL SBS (%) — Cross-model averages ──")
for lang in LANGUAGES:
    lang_df = stereo_df_out[stereo_df_out["language"] == lang]
    print(f"\n  {lang}:")
    for col in cat_cols:
        cat_name = col.replace("csbs_", "")
        avg = lang_df[col].mean()
        print(f"    {cat_name:30s}: {avg:.2f}%")

# ── Toxicity: STBS per subtype ────────────────────────────────────────────────
sub_cols = [c for c in toxicity_df_out.columns if c.startswith("stbs_")]
print(f"\n── SUBTYPE-LEVEL TBS — Cross-model averages ──")
for lang in LANGUAGES:
    lang_df = toxicity_df_out[toxicity_df_out["language"] == lang]
    print(f"\n  {lang}:")
    for col in sub_cols:
        sub_name = col.replace("stbs_", "").replace("_", " ").title()
        avg = lang_df[col].mean()
        print(f"    {sub_name:30s}: {avg:.6f}")

print("\n" + "="*70)
print(" Pipeline complete. All outputs saved to:", OUTPUT_DIR)
print("="*70)

# ─────────────────────────────────────────────────────────────────────────────
# CELL 8: Optional — Run single model or single language (for debugging/testing)
# ─────────────────────────────────────────────────────────────────────────────
# Uncomment to test just one model on English before running the full pipeline:

# stereo_scores_test, _ = run_stereotype_evaluation(
#     model_names=["BLOOM-560M"],
#     languages=["English"],
# )
# print(stereo_scores_test)

# tox_scores_test, _ = run_toxicity_evaluation(
#     model_names=["BLOOM-560M"],
#     languages=["English"],
# )
# print(tox_scores_test)
