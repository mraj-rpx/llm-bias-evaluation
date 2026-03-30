import os, sys, gc, re, logging
import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.config import (
    OUTPUT_DIR, STEREO_COL_MORE, STEREO_COL_LESS, STEREO_COL_CAT,
    TOXICITY_COL_PROMPT, NLLB_MODEL_ID, NLLB_CACHE_DIR,
    NLLB_SRC_LANG, NLLB_HI_LANG, NLLB_TA_LANG,
    TRANSLATION_BATCH_SIZE, TRANSLATION_MAX_LENGTH, TEST_RUN_LIMIT,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [translate] %(message)s")
log = logging.getLogger(__name__)

def remove_artifacts(text):
    if not isinstance(text, str): return ""
    text = re.sub(r"\s+", " ", text).strip()
    return text

class NLLBTranslator:
    def __init__(self, tgt_lang):
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        lang_name = "Hindi" if tgt_lang == NLLB_HI_LANG else "Tamil"
        log.info(f"Loading NLLB for English -> {lang_name}...")
        self.tgt_lang = tgt_lang
        self.tokenizer = AutoTokenizer.from_pretrained(
            NLLB_MODEL_ID, cache_dir=NLLB_CACHE_DIR, src_lang=NLLB_SRC_LANG)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            NLLB_MODEL_ID, cache_dir=NLLB_CACHE_DIR,
            use_safetensors=True, torch_dtype=torch.float16).cuda()
        self.model.eval()
        self.forced_bos_token_id = self.tokenizer.convert_tokens_to_ids(tgt_lang)
        log.info(f"  NLLB loaded for {lang_name}.")

    def translate_batch(self, sentences):
        inputs = self.tokenizer(sentences, return_tensors="pt", padding=True,
            truncation=True, max_length=TRANSLATION_MAX_LENGTH).to("cuda")
        with torch.no_grad():
            generated = self.model.generate(
                **inputs, forced_bos_token_id=self.forced_bos_token_id,
                num_beams=4, max_length=TRANSLATION_MAX_LENGTH)
        decoded = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
        return [remove_artifacts(t) for t in decoded]

    def translate_column(self, texts, desc="Translating"):
        results = []
        for i in tqdm(range(0, len(texts), TRANSLATION_BATCH_SIZE), desc=desc):
            batch = texts[i: i + TRANSLATION_BATCH_SIZE]
            try: translated = self.translate_batch(batch)
            except Exception as e:
                log.warning(f"  Batch {i} failed ({e}), using originals.")
                translated = batch
            results.extend(translated)
        return results

    def unload(self):
        del self.model, self.tokenizer
        gc.collect(); torch.cuda.empty_cache()
        log.info("  NLLB unloaded.")

def translate_datasets(stereo_df, toxicity_df):
    if TEST_RUN_LIMIT is not None:
        stereo_df   = stereo_df.head(TEST_RUN_LIMIT)
        toxicity_df = toxicity_df.head(TEST_RUN_LIMIT)
        log.info(f"TEST MODE: {TEST_RUN_LIMIT} rows only.")

    for df, name in [(stereo_df, "stereo"), (toxicity_df, "toxicity")]:
        out = df.copy(); out["language"] = "English"
        out.to_csv(os.path.join(OUTPUT_DIR, f"{name}_English.csv"), index=False)
    log.info("English CSVs saved.")

    for lang_name, tgt_code in [("Tamil", NLLB_TA_LANG), ("Hindi", NLLB_HI_LANG)]:
        s_out = os.path.join(OUTPUT_DIR, f"stereo_{lang_name}.csv")
        t_out = os.path.join(OUTPUT_DIR, f"toxicity_{lang_name}.csv")
        if os.path.exists(s_out) and os.path.exists(t_out):
            log.info(f"{lang_name} CSVs already exist — skipping."); continue

        tr = NLLBTranslator(tgt_lang=tgt_code)
        st = stereo_df.copy()
        st[STEREO_COL_MORE] = tr.translate_column(stereo_df[STEREO_COL_MORE].tolist(), f"sent_more->{lang_name}")
        st[STEREO_COL_LESS] = tr.translate_column(stereo_df[STEREO_COL_LESS].tolist(), f"sent_less->{lang_name}")
        st["language"] = lang_name; st.to_csv(s_out, index=False)
        log.info(f"  Saved: {s_out}")

        tx = toxicity_df.copy()
        tx[TOXICITY_COL_PROMPT] = tr.translate_column(toxicity_df[TOXICITY_COL_PROMPT].tolist(), f"prompts->{lang_name}")
        tx["language"] = lang_name; tx.to_csv(t_out, index=False)
        log.info(f"  Saved: {t_out}")
        tr.unload()

    log.info("All translations complete.")

if __name__ == "__main__":
    from modules.validate_data import run_validation
    ok1, ok2, sdf, tdf = run_validation()
    if ok1 and ok2: translate_datasets(sdf, tdf)