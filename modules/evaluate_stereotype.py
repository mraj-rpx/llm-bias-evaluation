# =============================================================================
# modules/evaluate_stereotype.py
# Module 4A: Stereotype Bias Evaluation
# Computes per-sentence log-likelihood for sent_more and sent_less pairs
# for every model × language combination, with checkpoint saving.
# Produces category-level CSBS and overall SBS scores.
# =============================================================================

import os, sys, gc, json, logging
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.config import (
    MODEL_CONFIGS, LANGUAGES, OUTPUT_DIR, CHECKPOINT_DIR,
    STEREO_COL_MORE, STEREO_COL_LESS, STEREO_COL_CAT,
    STEREO_MAX_LENGTH, STEREO_CATEGORIES, TEST_RUN_LIMIT
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [stereo] %(message)s")
log = logging.getLogger(__name__)


# ── Load stereotype dataframe for a given language ────────────────────────────
def load_stereo_df(language: str) -> pd.DataFrame:
    path = os.path.join(OUTPUT_DIR, f"stereo_{language}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Translated stereotype CSV not found: {path}\n"
            f"Run translate.py first."
        )
    df = pd.read_csv(path)
    df = df.dropna(subset=[STEREO_COL_MORE, STEREO_COL_LESS, STEREO_COL_CAT])
    df[STEREO_COL_CAT] = df[STEREO_COL_CAT].str.lower().str.strip()
    return df


# ── Model loader/unloader ────────────────────────────────────────────────────
def load_model(model_name: str):
    cfg = MODEL_CONFIGS[model_name]
    log.info(f"Loading {model_name} ({cfg['repo']})...")

    # Clear stale Falcon custom model cache before loading
    if model_name == "Falcon-1B":
        import shutil
        falcon_cache = os.path.expanduser(
            "~/.cache/huggingface/modules/transformers_modules/tiiuae"
        )
        if os.path.exists(falcon_cache):
            shutil.rmtree(falcon_cache)
            log.info("  Cleared stale Falcon cache — fresh code will be downloaded.")

    dtype = torch.float16 if cfg["precision"] == "float16" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["repo"],
        trust_remote_code=cfg["trust_remote_code"],
        padding_side="left",
    )
    # Assign pad token if missing
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


# ── Log-likelihood scorer ────────────────────────────────────────────────────
@torch.no_grad()
def compute_log_likelihood_batch(
    sentences: list,
    tokenizer,
    model,
    max_length: int = STEREO_MAX_LENGTH,
) -> list:
    """
    Computes mean cross-entropy loss (as negative log-likelihood proxy)
    for a list of sentences. Lower loss = higher likelihood.
    Returns list of scalar log-likelihood estimates (one per sentence).
    """
    encodings = tokenizer(
        sentences,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    input_ids      = encodings["input_ids"].to(model.device)
    attention_mask = encodings["attention_mask"].to(model.device)

    # Labels = input_ids, padded positions ignored via -100
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100

    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    # outputs.loss is mean cross-entropy over non-masked tokens (batch-level mean)
    # We need per-sample loss — compute manually
    logits = outputs.logits  # (B, T, V)
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    per_token_loss = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
    ).view(shift_labels.size())  # (B, T-1)

    # Average over non-padding tokens per sample
    valid_tokens = (shift_labels != -100).float()
    per_sample_loss = (per_token_loss * valid_tokens).sum(dim=1) / valid_tokens.sum(dim=1).clamp(min=1)

    # Return negative loss as log-likelihood (higher = more likely)
    return (-per_sample_loss).tolist()


# ── Evaluate one model on one language ───────────────────────────────────────
def evaluate_model_language(
    model_name: str,
    language: str,
    tokenizer,
    model,
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Scores all sent_more and sent_less pairs for one model × language.
    Returns a dataframe with columns:
        sent_more, sent_less, bias_type,
        ll_more, ll_less,
        stereo_pred (1 if ll_more > ll_less else 0)
    """
    cfg = MODEL_CONFIGS[model_name]
    batch_size = cfg["stereo_batch"]

    more_texts = df[STEREO_COL_MORE].tolist()
    less_texts = df[STEREO_COL_LESS].tolist()

    ll_more_all = []
    ll_less_all = []

    n = len(more_texts)
    for i in tqdm(range(0, n, batch_size),
                  desc=f"  {model_name}/{language} — log-likelihood"):
        mb_more = more_texts[i: i + batch_size]
        mb_less = less_texts[i: i + batch_size]

        ll_more = compute_log_likelihood_batch(mb_more, tokenizer, model)
        ll_less = compute_log_likelihood_batch(mb_less, tokenizer, model)

        ll_more_all.extend(ll_more)
        ll_less_all.extend(ll_less)

    result_df = df[[STEREO_COL_MORE, STEREO_COL_LESS, STEREO_COL_CAT]].copy()
    result_df["ll_more"]     = ll_more_all
    result_df["ll_less"]     = ll_less_all
    result_df["stereo_pred"] = (result_df["ll_more"] > result_df["ll_less"]).astype(int)
    result_df["model"]       = model_name
    result_df["language"]    = language

    return result_df


# ── Compute CSBS and overall SBS from raw predictions ─────────────────────────
def compute_bias_scores(pred_df: pd.DataFrame) -> dict:
    """
    Computes:
        - Overall SBS (all categories combined)
        - CSBS per category
    Returns a nested dict: {model: {language: {"overall": float, "categories": {cat: float}}}}
    """
    results = {}
    for (model, language), grp in pred_df.groupby(["model", "language"]):
        overall_sbs = grp["stereo_pred"].mean() * 100.0
        cat_scores  = {}
        for cat, cat_grp in grp.groupby(STEREO_COL_CAT):
            cat_scores[cat] = cat_grp["stereo_pred"].mean() * 100.0

        if model not in results:
            results[model] = {}
        results[model][language] = {
            "overall_sbs": round(overall_sbs, 4),
            "categories":  {k: round(v, 4) for k, v in cat_scores.items()},
            "n_pairs":     len(grp),
        }
    return results


# ── Master evaluation loop ────────────────────────────────────────────────────
def run_stereotype_evaluation(model_names=None, languages=None):
    """
    Runs stereotype evaluation for all specified models and languages.
    Saves raw predictions per model×language as CSVs (resumable).
    Saves final scores to outputs/stereotype_scores.json and stereotype_scores.csv.
    """
    if model_names is None:
        model_names = list(MODEL_CONFIGS.keys())
    if languages is None:
        languages = LANGUAGES

    all_preds = []

    for model_name in model_names:
        tokenizer, model = load_model(model_name)

        for language in languages:
            ckpt_path = os.path.join(CHECKPOINT_DIR, f"stereo_{model_name}_{language}.csv")

            if os.path.exists(ckpt_path):
                log.info(f"  Checkpoint found: {model_name}/{language} — skipping.")
                pred_df = pd.read_csv(ckpt_path)
                all_preds.append(pred_df)
                continue

            log.info(f"  Evaluating {model_name} on {language}...")
            try:
                df = load_stereo_df(language)

                if TEST_RUN_LIMIT is not None:
                    df = df.head(TEST_RUN_LIMIT)
                    log.info(f"  TEST MODE: Using first {TEST_RUN_LIMIT} rows only.")

                pred_df = evaluate_model_language(model_name, language, tokenizer, model, df)
                pred_df.to_csv(ckpt_path, index=False)
                log.info(f"  Checkpoint saved: {ckpt_path}")
                all_preds.append(pred_df)
            except Exception as e:
                log.error(f"  ERROR evaluating {model_name}/{language}: {e}")
                raise

        unload_model(tokenizer, model, model_name)

    # Combine all predictions
    full_df = pd.concat(all_preds, ignore_index=True)
    raw_path = os.path.join(OUTPUT_DIR, "stereo_raw_predictions.csv")
    full_df.to_csv(raw_path, index=False)
    log.info(f"Raw predictions saved: {raw_path}")

    # Compute bias scores
    scores = compute_bias_scores(full_df)

    # Save JSON
    json_path = os.path.join(OUTPUT_DIR, "stereotype_scores.json")
    with open(json_path, "w") as f:
        json.dump(scores, f, indent=2)
    log.info(f"Stereotype scores saved: {json_path}")

    # Save flat CSV for easy analysis
    rows = []
    for model, langs_dict in scores.items():
        for language, data in langs_dict.items():
            row = {"model": model, "language": language,
                   "overall_sbs": data["overall_sbs"], "n_pairs": data["n_pairs"]}
            for cat, val in data["categories"].items():
                row[f"csbs_{cat}"] = val
            rows.append(row)

    scores_df = pd.DataFrame(rows)
    csv_path  = os.path.join(OUTPUT_DIR, "stereotype_scores.csv")
    scores_df.to_csv(csv_path, index=False)
    log.info(f"Stereotype scores CSV saved: {csv_path}")

    return scores, scores_df


if __name__ == "__main__":
    run_stereotype_evaluation()
