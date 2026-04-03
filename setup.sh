#!/bin/bash
# =============================================================================
# setup.sh — Environment setup for RunPod PyTorch 2.4.0 template
# PyTorch is already installed in this template — skipping torch install
# =============================================================================

set -e
echo "============================================================"
echo " LLM Bias Evaluation Pipeline — Environment Setup"
echo " Template: RunPod PyTorch 2.4.0"
echo "============================================================"

# ── 1. Verify existing PyTorch installation ───────────────────────────────────
echo ""
echo "[1/5] Verifying PyTorch..."
python3 -c "
import torch
print(f'   PyTorch version : {torch.__version__}')
print(f'   CUDA version    : {torch.version.cuda}')
print(f'   CUDA available  : {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'   GPU             : {torch.cuda.get_device_name(0)}')
    print(f'   VRAM            : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
else:
    print('   ERROR: CUDA not available — wrong template selected')
    exit(1)
"

# ── 2. Install remaining packages from PyPI ───────────────────────────────────
echo ""
echo "[2/5] Installing pipeline packages..."
pip install -q \
    "transformers>=4.44.0" \
    "tokenizers>=0.19.0" \
    accelerate \
    sentencepiece \
    sacremoses \
    datasets \
    pandas \
    numpy \
    tqdm \
    detoxify \
    matplotlib \
    seaborn \
    ipywidgets \
    notebook \
    huggingface_hub

echo "   Packages installed."

# ── 3. Install IndicTrans2 ────────────────────────────────────────────────────
echo ""
echo "[3/5] IndicTrans2 skipped — using NLLB for both Tamil and Hindi."
pip install -q nltk sacrebleu

# ── 4. Download NLLB model (Hindi translation) ────────────────────────────────
echo ""
echo "[4/5] Downloading NLLB model for Hindi translation..."
python3 -c "
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
print('   Downloading NLLB tokenizer...')
AutoTokenizer.from_pretrained(
    'facebook/nllb-200-distilled-600M',
    cache_dir='models/nllb'
)
print('   Downloading NLLB model...')
AutoModelForSeq2SeqLM.from_pretrained(
    'facebook/nllb-200-distilled-600M',
    cache_dir='models/nllb',
    use_safetensors=True
)
print('   NLLB ready.')
"

# ── 5. HuggingFace login ──────────────────────────────────────────────────────
echo ""
echo "[5/5] HuggingFace login..."
echo "   Logging in with your HF token..."
echo "   (Required for LLaMA-2-7B access)"
huggingface-cli login

# ── Final verification ────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo " Verification"
echo "============================================================"
python3 -c "
import torch
from transformers import AutoTokenizer
from IndicTransToolkit.processor import IndicProcessor
from detoxify import Detoxify
import pandas, numpy, matplotlib
print('   torch        :', torch.__version__)
print('   transformers : OK')
print('   IndicTrans   : OK')
print('   detoxify     : OK')
print('   pandas       : OK')
print('   GPU ready    :', torch.cuda.is_available())
"

echo ""
echo "============================================================"
echo " Setup complete."
echo " Next steps:"
echo "   1. Copy your CSVs to data/"
echo "   2. jupyter notebook pipeline_notebook.ipynb"
echo "============================================================"