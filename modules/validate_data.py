# =============================================================================
# modules/validate_data.py
# Validates both input CSVs before the pipeline starts.
# Run this first in the notebook to catch column name issues early.
# =============================================================================

import pandas as pd
import sys
import os

# Allow import from parent directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.config import (
    STEREOTYPE_CSV, TOXICITY_CSV,
    STEREO_COL_MORE, STEREO_COL_LESS, STEREO_COL_CAT,
    TOXICITY_COL_PROMPT, STEREO_CATEGORIES,
)


def validate_stereotype_csv(verbose=True):
    """
    Validates stereo_type_dataset.csv.
    Expected columns: sent_more, sent_less, bias_type
    Returns: (ok: bool, df: DataFrame or None)
    """
    print("=" * 60)
    print(" Validating stereo_type_dataset.csv")
    print("=" * 60)

    if not os.path.exists(STEREOTYPE_CSV):
        print(f"  ERROR: File not found: {STEREOTYPE_CSV}")
        print("  Place your stereo_type_dataset.csv in the data/ folder.")
        return False, None

    df = pd.read_csv(STEREOTYPE_CSV)
    print(f"  Rows: {len(df):,}")
    print(f"  Columns found: {list(df.columns)}")

    ok = True

    # Check required columns (case-insensitive)
    col_map = {c.lower(): c for c in df.columns}
    required = {
        STEREO_COL_MORE: "stereotypical sentence",
        STEREO_COL_LESS: "anti-stereotypical sentence",
        STEREO_COL_CAT:  "bias category label",
    }

    for expected, description in required.items():
        if expected.lower() in col_map:
            actual = col_map[expected.lower()]
            if actual != expected:
                print(f"  NOTE: Column '{actual}' will be used as '{expected}' ({description})")
                df = df.rename(columns={actual: expected})
            else:
                print(f"  OK  : Column '{expected}' found ({description})")
        else:
            print(f"  ERROR: Missing column '{expected}' ({description})")
            print(f"         Available columns: {list(df.columns)}")
            ok = False

    if not ok:
        print("\n  ACTION REQUIRED: Rename the columns in your CSV to match the expected names above.")
        return False, None

    # Check for nulls
    for col in [STEREO_COL_MORE, STEREO_COL_LESS, STEREO_COL_CAT]:
        n_null = df[col].isna().sum()
        if n_null > 0:
            print(f"  WARNING: {n_null} null values in column '{col}' — these rows will be dropped.")
            df = df.dropna(subset=[col])

    # Check categories
    found_cats = df[STEREO_COL_CAT].str.lower().str.strip().unique().tolist()
    print(f"\n  Bias categories found ({len(found_cats)}):")
    for cat in sorted(found_cats):
        n = len(df[df[STEREO_COL_CAT].str.lower().str.strip() == cat])
        expected_flag = "✓" if cat in STEREO_CATEGORIES else "?"
        print(f"    {expected_flag} {cat}: {n:,} pairs")

    if verbose:
        print("\n  Sample rows:")
        print(df[[STEREO_COL_MORE, STEREO_COL_LESS, STEREO_COL_CAT]].head(3).to_string(index=False))

    print(f"\n  RESULT: Stereotype CSV is {'VALID' if ok else 'INVALID'}.")
    return ok, df


def validate_toxicity_csv(verbose=True):
    """
    Validates toxicity.csv.
    Expected column: prompt
    Returns: (ok: bool, df: DataFrame or None)
    """
    print()
    print("=" * 60)
    print(" Validating toxicity.csv")
    print("=" * 60)

    if not os.path.exists(TOXICITY_CSV):
        print(f"  ERROR: File not found: {TOXICITY_CSV}")
        print("  Place your toxicity.csv in the data/ folder.")
        return False, None

    df = pd.read_csv(TOXICITY_CSV)
    print(f"  Rows: {len(df):,}")
    print(f"  Columns found: {list(df.columns)}")

    ok = True
    col_map = {c.lower(): c for c in df.columns}

    if TOXICITY_COL_PROMPT.lower() in col_map:
        actual = col_map[TOXICITY_COL_PROMPT.lower()]
        if actual != TOXICITY_COL_PROMPT:
            print(f"  NOTE: Column '{actual}' will be used as '{TOXICITY_COL_PROMPT}'")
            df = df.rename(columns={actual: TOXICITY_COL_PROMPT})
        else:
            print(f"  OK  : Column '{TOXICITY_COL_PROMPT}' found")
    else:
        # Try to find any text-like column
        text_candidates = [c for c in df.columns
                           if df[c].dtype == object and df[c].str.len().mean() > 20]
        if text_candidates:
            print(f"  NOTE: Column '{TOXICITY_COL_PROMPT}' not found.")
            print(f"        Candidate text columns: {text_candidates}")
            print(f"        Update TOXICITY_COL_PROMPT in config.py to match.")
        else:
            print(f"  ERROR: Column '{TOXICITY_COL_PROMPT}' not found.")
        ok = False

    if not ok:
        return False, None

    # Drop nulls and empty strings
    n_before = len(df)
    df = df.dropna(subset=[TOXICITY_COL_PROMPT])
    df = df[df[TOXICITY_COL_PROMPT].str.strip().str.len() > 0]
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        print(f"  WARNING: {n_dropped} null/empty prompt rows dropped.")

    print(f"  Valid prompts: {len(df):,}")

    if verbose:
        print("\n  Sample prompts:")
        for p in df[TOXICITY_COL_PROMPT].head(3).tolist():
            print(f"    '{p[:100]}'")

    print(f"\n  RESULT: Toxicity CSV is {'VALID' if ok else 'INVALID'}.")
    return ok, df


def run_validation():
    """Run both validations. Returns (stereo_ok, tox_ok, stereo_df, tox_df)."""
    stereo_ok, stereo_df = validate_stereotype_csv(verbose=True)
    tox_ok, tox_df       = validate_toxicity_csv(verbose=True)

    print()
    print("=" * 60)
    print(" VALIDATION SUMMARY")
    print("=" * 60)
    print(f"  Stereotype CSV : {'PASS' if stereo_ok else 'FAIL'}")
    print(f"  Toxicity CSV   : {'PASS' if tox_ok else 'FAIL'}")

    if stereo_ok and tox_ok:
        print("\n  Both CSVs are valid. You can proceed to the next step.")
    else:
        print("\n  Fix the errors above before running the pipeline.")

    return stereo_ok, tox_ok, stereo_df, tox_df


if __name__ == "__main__":
    run_validation()
