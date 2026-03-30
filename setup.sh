#!/bin/bash
# =============================================================================
# setup.sh — Environment setup for RunPod PyTorch 2.1.0 template
# PyTorch is already installed in this template — skipping torch install
# =============================================================================

set -e
echo "============================================================"
echo " LLM Bias Evaluation Pipeline — Environment Setup"
echo " Template: RunPod PyTorch 2.1.0"
echo "============================================================"

# ── 1. Verify existing PyTorch installation ───────────────────────────────────
echo ""
echo "[1/5] Verifying PyTorch..."
python3 -c "
import torch
print(f'   PyTorch version : {torch.__version__}')
print(f'   CUDA available  : {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'   GPU             : {torch.cuda.get_device_name(0)}')
    print(f'   VRAM            : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"

# ── 2. Install remaining packages from PyPI ───────────────────────────────────
echo ""
echo "[2/5] Installing pipeline packages..."
pip install -q \
    transformers \
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
    indic-nlp-library \
    IndicTransToolkit \
    huggingface_hub

echo "   Packages installed."

# ── 3. Install IndicTrans2 ────────────────────────────────────────────────────
echo ""
echo "[3/5] Installing IndicTrans2..."
if [ ! -d "IndicTrans2" ]; then
    git clone https://github.com/AI4Bharat/IndicTrans2.git
fi

pip install -q nltk sacrebleu

# Verify IndicTransToolkit import
python3 -c "
try:
    from IndicTransToolkit.processor import IndicProcessor
    print('   IndicTransToolkit ready.')
except ImportError as e:
    print(f'   WARNING: {e}')
    print('   Trying fallback install...')
" || true

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
    cache_dir='models/nllb'
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