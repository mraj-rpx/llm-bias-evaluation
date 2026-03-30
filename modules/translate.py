# =============================================================================
# modules/translate.py
# Module 2: Multilingual Translation
# Translates English stereotype pairs and toxicity prompts into Tamil (IndicTrans2)
# and Hindi (NLLB). Saves translated CSVs to outputs/ with checkpointing.
# =============================================================================

import os, sys, gc, re, logging
import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.config import (
    OUTPUT_DIR, CHECKPOINT_DIR,
    STEREO_COL_MORE, STEREO_COL_LESS, STEREO_COL_CAT,
    TOXICITY_COL_PROMPT,
    INDICTRANS2_DIR, NLLB_MODEL_ID, NLLB_CACHE_DIR,
    NLLB_SRC_LANG, NLLB_HI_LANG,
    TRANSLATION_BATCH_SIZE, TRANSLATION_MAX_LENGTH,
)
from modules.config import TEST_RUN_LIMIT  # ← ADD THIS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [translate] %(message)s")
log = logging.getLogger(__name__)


# ── Artifact removal ──────────────────────────────────────────────────────────
def remove_artifacts(text: str) -> str:
    """Remove common translation artifacts: extra spaces, repeated punctuation."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"([।!?])\1+", r"\1", text)   # deduplicate sentence-end markers
    text = re.sub(r"\.{4,}", "...", text)
    return text


# ── IndicTrans2 (English → Tamil) ────────────────────────────────────────────
class IndicTrans2Translator:
    def __init__(self):
        log.info("Loading IndicTrans2 model for English → Tamil...")
        try:
            from IndicTransToolkit.processor import IndicProcessor
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

            self.ip = IndicProcessor(inference=True)
            self.tokenizer = AutoTokenizer.from_pretrained(
                INDICTRANS2_DIR, trust_remote_code=True
            )
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                INDICTRANS2_DIR,
                trust_remote_code=True,
                torch_dtype=torch.float16,
            ).cuda()
            self.model.eval()
            log.info("IndicTrans2 loaded.")
        except ImportError:
            raise ImportError(
                "IndicTransToolkit not found. Run setup.sh first to install IndicTrans2."
            )

    def translate_batch(self, sentences: list, src_lang="eng_Latn", tgt_lang="hin_Deva"):
        # For Tamil, tgt_lang should be "tam_Taml"
        tgt_lang = "tam_Taml"
        batch = self.ip.preprocess_batch(sentences, src_lang=src_lang, tgt_lang=tgt_lang)
        inputs = self.tokenizer(
            batch,
            truncation=True,
            padding="longest",
            return_tensors="pt",
            max_length=TRANSLATION_MAX_LENGTH,
            return_attention_mask=True,
        ).to("cuda")
        with torch.no_grad():
            generated = self.model.generate(
                **inputs,
                num_beams=4,
                num_return_sequences=1,
                max_length=TRANSLATION_MAX_LENGTH,
            )
        decoded = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
        postprocessed = self.ip.postprocess_batch(decoded, lang=tgt_lang)
        return [remove_artifacts(t) for t in postprocessed]

    def translate_column(self, texts: list, desc="Translating to Tamil") -> list:
        results = []
        for i in tqdm(range(0, len(texts), TRANSLATION_BATCH_SIZE), desc=desc):
            batch = texts[i: i + TRANSLATION_BATCH_SIZE]
            try:
                translated = self.translate_batch(batch)
            except Exception as e:
                log.warning(f"  Batch {i}–{i+len(batch)} failed ({e}), using originals.")
                translated = batch
            results.extend(translated)
        return results

    def unload(self):
        del self.model, self.tokenizer, self.ip
        gc.collect()
        torch.cuda.empty_cache()
        log.info("IndicTrans2 unloaded.")


# ── NLLB (English → Hindi) ───────────────────────────────────────────────────
class NLLBTranslator:
    def __init__(self):
        log.info("Loading NLLB model for English → Hindi...")
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            NLLB_MODEL_ID, cache_dir=NLLB_CACHE_DIR, src_lang=NLLB_SRC_LANG
        )
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            NLLB_MODEL_ID,
            cache_dir=NLLB_CACHE_DIR,
            torch_dtype=torch.float16,
        ).cuda()
        self.model.eval()
        self.forced_bos_token_id = self.tokenizer.convert_tokens_to_ids(NLLB_HI_LANG)
        log.info("NLLB loaded.")

    def translate_batch(self, sentences: list) -> list:
        inputs = self.tokenizer(
            sentences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=TRANSLATION_MAX_LENGTH,
        ).to("cuda")
        with torch.no_grad():
            generated = self.model.generate(
                **inputs,
                forced_bos_token_id=self.forced_bos_token_id,
                num_beams=4,
                max_length=TRANSLATION_MAX_LENGTH,
            )
        decoded = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
        return [remove_artifacts(t) for t in decoded]

    def translate_column(self, texts: list, desc="Translating to Hindi") -> list:
        results = []
        for i in tqdm(range(0, len(texts), TRANSLATION_BATCH_SIZE), desc=desc):
            batch = texts[i: i + TRANSLATION_BATCH_SIZE]
            try:
                translated = self.translate_batch(batch)
            except Exception as e:
                log.warning(f"  Batch {i}–{i+len(batch)} failed ({e}), using originals.")
                translated = batch
            results.extend(translated)
        return results

    def unload(self):
        del self.model, self.tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        log.info("NLLB unloaded.")


# ── Main translation function ─────────────────────────────────────────────────
def translate_datasets(stereo_df: pd.DataFrame, toxicity_df: pd.DataFrame):
    if TEST_RUN_LIMIT is not None:              # ← ADD THIS
        stereo_df   = stereo_df.head(TEST_RUN_LIMIT)
        toxicity_df = toxicity_df.head(TEST_RUN_LIMIT)
        log.info(f"TEST MODE: Translating first {TEST_RUN_LIMIT} rows only.")

    """
    Translates both datasets into Tamil and Hindi.
    Saves 4 output CSVs and returns file paths.
    Checkpoints are saved after each language so the run is resumable.

    Output files:
        outputs/stereo_Tamil.csv
        outputs/stereo_Hindi.csv
        outputs/toxicity_Tamil.csv
        outputs/toxicity_Hindi.csv
    """
    out_paths = {
        "stereo_Tamil":   os.path.join(OUTPUT_DIR, "stereo_Tamil.csv"),
        "stereo_Hindi":   os.path.join(OUTPUT_DIR, "stereo_Hindi.csv"),
        "toxicity_Tamil": os.path.join(OUTPUT_DIR, "toxicity_Tamil.csv"),
        "toxicity_Hindi": os.path.join(OUTPUT_DIR, "toxicity_Hindi.csv"),
    }

    # ── TAMIL (IndicTrans2) ───────────────────────────────────────────────────
    if os.path.exists(out_paths["stereo_Tamil"]) and os.path.exists(out_paths["toxicity_Tamil"]):
        log.info("Tamil CSVs already exist. Skipping Tamil translation.")
    else:
        log.info("Starting Tamil translation with IndicTrans2...")
        ta_translator = IndicTrans2Translator()

        # Stereotype
        log.info("  Translating stereotype pairs to Tamil...")
        stereo_ta = stereo_df.copy()
        stereo_ta[STEREO_COL_MORE] = ta_translator.translate_column(
            stereo_df[STEREO_COL_MORE].tolist(), "sent_more → Tamil"
        )
        stereo_ta[STEREO_COL_LESS] = ta_translator.translate_column(
            stereo_df[STEREO_COL_LESS].tolist(), "sent_less → Tamil"
        )
        stereo_ta["language"] = "Tamil"
        stereo_ta.to_csv(out_paths["stereo_Tamil"], index=False)
        log.info(f"  Saved: {out_paths['stereo_Tamil']}")

        # Toxicity
        log.info("  Translating toxicity prompts to Tamil...")
        tox_ta = toxicity_df.copy()
        tox_ta[TOXICITY_COL_PROMPT] = ta_translator.translate_column(
            toxicity_df[TOXICITY_COL_PROMPT].tolist(), "prompts → Tamil"
        )
        tox_ta["language"] = "Tamil"
        tox_ta.to_csv(out_paths["toxicity_Tamil"], index=False)
        log.info(f"  Saved: {out_paths['toxicity_Tamil']}")

        ta_translator.unload()

    # ── HINDI (NLLB) ─────────────────────────────────────────────────────────
    if os.path.exists(out_paths["stereo_Hindi"]) and os.path.exists(out_paths["toxicity_Hindi"]):
        log.info("Hindi CSVs already exist. Skipping Hindi translation.")
    else:
        log.info("Starting Hindi translation with NLLB...")
        hi_translator = NLLBTranslator()

        # Stereotype
        log.info("  Translating stereotype pairs to Hindi...")
        stereo_hi = stereo_df.copy()
        stereo_hi[STEREO_COL_MORE] = hi_translator.translate_column(
            stereo_df[STEREO_COL_MORE].tolist(), "sent_more → Hindi"
        )
        stereo_hi[STEREO_COL_LESS] = hi_translator.translate_column(
            stereo_df[STEREO_COL_LESS].tolist(), "sent_less → Hindi"
        )
        stereo_hi["language"] = "Hindi"
        stereo_hi.to_csv(out_paths["stereo_Hindi"], index=False)
        log.info(f"  Saved: {out_paths['stereo_Hindi']}")

        # Toxicity
        log.info("  Translating toxicity prompts to Hindi...")
        tox_hi = toxicity_df.copy()
        tox_hi[TOXICITY_COL_PROMPT] = hi_translator.translate_column(
            toxicity_df[TOXICITY_COL_PROMPT].tolist(), "prompts → Hindi"
        )
        tox_hi["language"] = "Hindi"
        tox_hi.to_csv(out_paths["toxicity_Hindi"], index=False)
        log.info(f"  Saved: {out_paths['toxicity_Hindi']}")

        hi_translator.unload()

    # Add English originals (with language tag)
    stereo_en = stereo_df.copy()
    stereo_en["language"] = "English"
    stereo_en.to_csv(os.path.join(OUTPUT_DIR, "stereo_English.csv"), index=False)

    tox_en = toxicity_df.copy()
    tox_en["language"] = "English"
    tox_en.to_csv(os.path.join(OUTPUT_DIR, "toxicity_English.csv"), index=False)

    log.info("Translation complete. All 6 dataset files saved.")
    return out_paths


if __name__ == "__main__":
    from modules.validate_data import run_validation
    stereo_ok, tox_ok, stereo_df, tox_df = run_validation()
    if stereo_ok and tox_ok:
        translate_datasets(stereo_df, tox_df)
