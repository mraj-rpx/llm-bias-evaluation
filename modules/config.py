# =============================================================================
# modules/config.py
# Central configuration for the LLM Bias Evaluation Pipeline
# All modules import from here. Change settings here only.
# =============================================================================

import os

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR         = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR         = os.path.join(BASE_DIR, "data")
OUTPUT_DIR       = os.path.join(BASE_DIR, "outputs")
CHECKPOINT_DIR   = os.path.join(BASE_DIR, "checkpoints")
LOG_DIR          = os.path.join(BASE_DIR, "logs")
MODEL_CACHE_DIR  = os.path.join(BASE_DIR, "models")

# ── Input CSV file paths ──────────────────────────────────────────────────────
STEREOTYPE_CSV   = os.path.join(DATA_DIR, "stereo_type_dataset.csv")
TOXICITY_CSV     = os.path.join(DATA_DIR, "toxicity.csv")

# ── Expected CSV columns ──────────────────────────────────────────────────────
# stereo_type_dataset.csv must have these columns (case-insensitive check done at runtime):
STEREO_COL_MORE  = "sent_more"      # stereotypical sentence
STEREO_COL_LESS  = "sent_less"      # anti-stereotypical sentence
STEREO_COL_CAT   = "bias_type"      # category label (race, gender, religion, etc.)

# toxicity.csv must have this column:
TOXICITY_COL_PROMPT = "prompt"      # text prompt for continuation

# ── Test-run limit (set to None for full dataset, or an integer e.g. 10) ──────
TEST_RUN_LIMIT = None                  # ← ADD THIS LINE

# ── Languages ─────────────────────────────────────────────────────────────────
LANGUAGES        = ["English", "Tamil", "Hindi"]

# ── Translation model paths ───────────────────────────────────────────────────
INDICTRANS2_DIR  = os.path.join(BASE_DIR, "models", "indictrans2-en-indic")
NLLB_MODEL_ID    = "facebook/nllb-200-distilled-600M"
NLLB_CACHE_DIR   = os.path.join(BASE_DIR, "models", "nllb")

# ── NLLB language codes ───────────────────────────────────────────────────────
NLLB_SRC_LANG    = "eng_Latn"
NLLB_HI_LANG     = "hin_Deva"
NLLB_TA_LANG     = "tam_Taml"

# ── Translation batch sizes ───────────────────────────────────────────────────
TRANSLATION_BATCH_SIZE = 32   # number of sentences per translation batch
TRANSLATION_MAX_LENGTH = 256  # max tokens for translated output

# ── LLM model configurations ─────────────────────────────────────────────────
MODEL_CONFIGS = {
    "LLaMA-2-7B": {
        "repo":        "NousResearch/Llama-2-7b-hf",
        "precision":   "float16",
        "device_map":  "auto",
        "trust_remote_code": False,
        "stereo_batch": 4,
        "toxicity_batch": 4,
    },
    "BLOOM-560M": {
        "repo":        "bigscience/bloom-560m",
        "precision":   "float32",   # float16 causes NaN in log-likelihood
        "device_map":  "cuda",
        "trust_remote_code": False,
        "stereo_batch": 64,
        "toxicity_batch": 8,
    },
    "Falcon-1B": {
        "repo":        "tiiuae/falcon-rw-1b",
        "precision":   "float16",
        "device_map":  "cuda",
        "trust_remote_code": True,  # required for Falcon custom architecture
        "stereo_batch": 32,
        "toxicity_batch": 8,
    },
    "Mistral-7B": {
        "repo":        "mistralai/Mistral-7B-v0.1",
        "precision":   "float16",
        "device_map":  "auto",
        "trust_remote_code": False,
        "stereo_batch": 16,
        "toxicity_batch": 8,
    },
    "GPT-J-6B": {
        "repo":        "EleutherAI/gpt-j-6b",
        "precision":   "float16",
        "device_map":  "auto",
        "trust_remote_code": False,
        "stereo_batch": 8,
        "toxicity_batch": 8,
    },
}

# ── Log-likelihood scoring settings ──────────────────────────────────────────
STEREO_MAX_LENGTH = 512     # max tokens for stereotype sentence pairs

# ── Text generation settings (toxicity) ───────────────────────────────────────
GEN_MAX_NEW_TOKENS = 50     # greedy decoding, max new tokens
GEN_DO_SAMPLE      = False  # greedy (deterministic)
GEN_TRUNC_LENGTH   = 512    # max input tokens (left-padded)

# ── Detoxify settings ─────────────────────────────────────────────────────────
DETOXIFY_EN_MODEL   = "original"       # for English continuations
DETOXIFY_MULTI_MODEL= "multilingual"   # for Tamil and Hindi continuations
DETOXIFY_BATCH_SIZE = 128

# ── CrowS-Pairs bias categories ───────────────────────────────────────────────
STEREO_CATEGORIES = [
    "race-color",
    "gender",
    "socioeconomic",
    "nationality",
    "religion",
    "age",
    "sexual-orientation",
    "physical-appearance",
    "disability",
]

# ── Detoxify output subtypes ──────────────────────────────────────────────────
TOXICITY_SUBTYPES = [
    "toxicity",
    "severe_toxicity",
    "obscene",
    "threat",
    "insult",
    "identity_attack",
]

# ── Ensure output directories exist ──────────────────────────────────────────
for d in [OUTPUT_DIR, CHECKPOINT_DIR, LOG_DIR]:
    os.makedirs(d, exist_ok=True)
