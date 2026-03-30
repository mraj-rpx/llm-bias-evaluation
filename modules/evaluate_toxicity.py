# =============================================================================
# modules/evaluate_toxicity.py
# Module 4B: Toxicity Bias Evaluation
# Generates text continuations for each prompt (greedy decoding) then scores
# all 6 Detoxify subtypes per continuation per model × language.
# Saves raw continuations, raw subtype scores, and aggregated STBS CSV/JSON.
# =============================================================================

import os, sys, gc, json, logging
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from detoxify import Detoxify

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.config import (
    MODEL_CONFIGS, LANGUAGES, OUTPUT_DIR, CHECKPOINT_DIR,
    TOXICITY_COL_PROMPT, TOXICITY_SUBTYPES,
    GEN_MAX_NEW_TOKENS, GEN_DO_SAMPLE, GEN_TRUNC_LENGTH,
    DETOXIFY_EN_MODEL, DETOXIFY_MULTI_MODEL, DETOXIFY_BATCH_SIZE,
    TEST_RUN_LIMIT
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [toxicity] %(message)s")
log = logging.getLogger(__name__)


# ── Load toxicity dataframe for a given language ──────────────────────────────
def load_toxicity_df(language: str) -> pd.DataFrame:
    path = os.path.join(OUTPUT_DIR, f"toxicity_{language}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Toxicity CSV not found: {path}\n"
            f"Run translate.py first."
        )
    df = pd.read_csv(path)
    df = df.dropna(subset=[TOXICITY_COL_PROMPT])
    df = df[df[TOXICITY_COL_PROMPT].str.strip().str.len() > 0]
    return df.reset_index(drop=True)


# ── Model loader/unloader ─────────────────────────────────────────────────────
def load_model(model_name: str):
    cfg = MODEL_CONFIGS[model_name]
    log.info(f"Loading {model_name} ({cfg['repo']})...")
    dtype = torch.float16 if cfg["precision"] == "float16" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["repo"],
        trust_remote_code=cfg["trust_remote_code"],
    use_safetensors=True,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        cfg["repo"],
        torch_dtype=dtype,
        device_map=cfg["device_map"],
        trust_remote_code=cfg["trust_remote_code"],
        use_safetensors=True,
        revision=cfg.get("revision", "main"),
    )
    model.eval()
    log.info(f"  {model_name} loaded.")
    return tokenizer, model


def unload_model(tokenizer, model, model_name):
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    log.info(f"  {model_name} unloaded and GPU memory cleared.")


# ── Text generation ───────────────────────────────────────────────────────────
@torch.no_grad()
def generate_continuations_batch(
    prompts: list,
    tokenizer,
    model,
) -> list:
    """
    Greedy-decodes a continuation for each prompt.
    Returns list of continuation strings (new tokens only, not including prompt).
    """
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=GEN_TRUNC_LENGTH,
    ).to(model.device)

    prompt_len = inputs["input_ids"].shape[1]

    generated = model.generate(
        **inputs,
        max_new_tokens=GEN_MAX_NEW_TOKENS,
        do_sample=GEN_DO_SAMPLE,
        pad_token_id=tokenizer.pad_token_id,
    )

    # Decode only the newly generated tokens (strip prompt prefix)
    new_tokens = generated[:, prompt_len:]
    continuations = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
    return continuations


def generate_all_continuations(
    model_name: str,
    language: str,
    tokenizer,
    model,
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Generates continuations for all prompts. Returns DataFrame with
    columns: prompt, continuation, model, language
    """
    cfg       = MODEL_CONFIGS[model_name]
    batch_sz  = cfg["toxicity_batch"]
    prompts   = df[TOXICITY_COL_PROMPT].tolist()
    continuations = []

    for i in tqdm(range(0, len(prompts), batch_sz),
                  desc=f"  {model_name}/{language} — generating"):
        batch = prompts[i: i + batch_sz]
        try:
            cont = generate_continuations_batch(batch, tokenizer, model)
        except Exception as e:
            log.warning(f"  Generation batch {i} failed ({e}). Using empty strings.")
            cont = [""] * len(batch)
        continuations.extend(cont)

    result = df[[TOXICITY_COL_PROMPT]].copy()
    result["continuation"] = continuations
    result["model"]        = model_name
    result["language"]     = language
    return result


# ── Detoxify scoring ──────────────────────────────────────────────────────────
def score_continuations(cont_df: pd.DataFrame, language: str) -> pd.DataFrame:
    """
    Scores all continuations with Detoxify.
    English → original Detoxify model
    Tamil/Hindi → multilingual Detoxify model
    Returns the input DataFrame with 6 new subtype columns appended.
    """
    model_type = DETOXIFY_EN_MODEL if language == "English" else DETOXIFY_MULTI_MODEL
    log.info(f"  Scoring {len(cont_df):,} continuations with Detoxify ({model_type})...")

    detector = Detoxify(model_type)
    texts    = cont_df["continuation"].fillna("").tolist()

    all_scores = {sub: [] for sub in TOXICITY_SUBTYPES}

    for i in tqdm(range(0, len(texts), DETOXIFY_BATCH_SIZE),
                  desc=f"  Detoxify/{language}"):
        batch = texts[i: i + DETOXIFY_BATCH_SIZE]
        # Handle empty strings gracefully
        batch_safe = [t if t.strip() else "." for t in batch]
        try:
            results = detector.predict(batch_safe)
        except Exception as e:
            log.warning(f"  Detoxify batch {i} failed ({e}). Using zeros.")
            results = {sub: [0.0] * len(batch) for sub in TOXICITY_SUBTYPES}

        for sub in TOXICITY_SUBTYPES:
            vals = results.get(sub, [0.0] * len(batch))
            # detoxify may return a tensor or list
            if hasattr(vals, "tolist"):
                vals = vals.tolist()
            all_scores[sub].extend(vals)

    del detector
    gc.collect()

    scored = cont_df.copy()
    for sub in TOXICITY_SUBTYPES:
        scored[sub] = all_scores[sub]
    return scored


# ── Compute STBS from raw subtype scores ─────────────────────────────────────
def compute_stbs(scored_df: pd.DataFrame) -> dict:
    """
    Computes Subtype-Level Toxicity Bias Score (STBS) = mean per subtype.
    Returns nested dict: {model: {language: {subtype: float}}}
    """
    results = {}
    for (model, language), grp in scored_df.groupby(["model", "language"]):
        subtype_means = {}
        for sub in TOXICITY_SUBTYPES:
            if sub in grp.columns:
                subtype_means[sub] = round(float(grp[sub].mean()), 6)
            else:
                subtype_means[sub] = None
        if model not in results:
            results[model] = {}
        results[model][language] = {
            "subtypes": subtype_means,
            "n_prompts": len(grp),
        }
    return results


# ── Master evaluation loop ────────────────────────────────────────────────────
def run_toxicity_evaluation(model_names=None, languages=None):
    """
    Runs toxicity evaluation for all models and languages.
    Step 1: Generate continuations (saved as checkpoints per model×language)
    Step 2: Score with Detoxify (appended to continuation checkpoints)
    Step 3: Aggregate STBS and save results.
    """
    if model_names is None:
        model_names = list(MODEL_CONFIGS.keys())
    if languages is None:
        languages = LANGUAGES

    all_scored = []

    # ── Step 1 + 2: Generate and score ───────────────────────────────────────
    for model_name in model_names:
        tokenizer, model = load_model(model_name)

        for language in languages:
            scored_ckpt = os.path.join(
                CHECKPOINT_DIR, f"toxicity_scored_{model_name}_{language}.csv"
            )

            if os.path.exists(scored_ckpt):
                log.info(f"  Checkpoint found: {model_name}/{language} — skipping generation.")
                scored_df = pd.read_csv(scored_ckpt)
                all_scored.append(scored_df)
                continue

            # Check if continuations already generated (but not yet scored)
            cont_ckpt = os.path.join(
                CHECKPOINT_DIR, f"toxicity_cont_{model_name}_{language}.csv"
            )

            if os.path.exists(cont_ckpt):
                log.info(f"  Continuations found for {model_name}/{language}, skipping generation.")
                cont_df = pd.read_csv(cont_ckpt)
            else:
                log.info(f"  Generating continuations: {model_name}/{language}...")
                tox_df  = load_toxicity_df(language)

                if TEST_RUN_LIMIT is not None:
                    tox_df = tox_df.head(TEST_RUN_LIMIT)
                    log.info(f"  TEST MODE: Using first {TEST_RUN_LIMIT} rows only.")

                cont_df = generate_all_continuations(model_name, language, tokenizer, model, tox_df)
                cont_df.to_csv(cont_ckpt, index=False)
                log.info(f"  Continuations saved: {cont_ckpt}")

            # Score with Detoxify (don't need LLM for this)
            scored_df = score_continuations(cont_df, language)
            scored_df.to_csv(scored_ckpt, index=False)
            log.info(f"  Scored checkpoint saved: {scored_ckpt}")
            all_scored.append(scored_df)

        unload_model(tokenizer, model, model_name)

    # ── Step 3: Aggregate ─────────────────────────────────────────────────────
    full_df  = pd.concat(all_scored, ignore_index=True)
    raw_path = os.path.join(OUTPUT_DIR, "toxicity_raw_scored.csv")
    full_df.to_csv(raw_path, index=False)
    log.info(f"Raw scored continuations saved: {raw_path}")

    stbs = compute_stbs(full_df)

    json_path = os.path.join(OUTPUT_DIR, "toxicity_scores.json")
    with open(json_path, "w") as f:
        json.dump(stbs, f, indent=2)
    log.info(f"Toxicity scores saved: {json_path}")

    # Flat CSV
    rows = []
    for model, langs_dict in stbs.items():
        for language, data in langs_dict.items():
            row = {"model": model, "language": language, "n_prompts": data["n_prompts"]}
            for sub, val in data["subtypes"].items():
                row[f"stbs_{sub}"] = val
            rows.append(row)
    scores_df = pd.DataFrame(rows)
    csv_path  = os.path.join(OUTPUT_DIR, "toxicity_scores.csv")
    scores_df.to_csv(csv_path, index=False)
    log.info(f"Toxicity scores CSV saved: {csv_path}")

    return stbs, scores_df


if __name__ == "__main__":
    run_toxicity_evaluation()
