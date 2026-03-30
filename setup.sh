#!/bin/bash
# =============================================================================
# setup.sh — Environment setup for the LLM Bias Evaluation Pipeline
# Run ONCE on a fresh RunPod instance before starting the notebook
# =============================================================================

set -e
echo "============================================================"
echo " LLM Bias Evaluation Pipeline — Environment Setup"
echo "============================================================"

# ── 1. Core Python packages ───────────────────────────────────────────────────
echo ""
echo "[1/5] Installing core packages..."
pip install -q \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 \
    transformers==4.40.0 \
    accelerate==0.29.3 \
    sentencepiece==0.2.0 \
    sacremoses \
    datasets \
    pandas \
    numpy \
    tqdm \
    detoxify \
    matplotlib \
    seaborn \
    ipywidgets \
    jupyter \
    notebook

echo "   Core packages installed."

# ── 2. IndicTrans2 (AI4Bharat) for Tamil translation ─────────────────────────
echo ""
echo "[2/5] Installing IndicTrans2 for Tamil translation..."
pip install -q \
    nltk \
    sacrebleu \
    indic-nlp-library

# Clone and install IndicTrans2
if [ ! -d "IndicTrans2" ]; then
    git clone https://github.com/AI4Bharat/IndicTrans2.git
fi
cd IndicTrans2
pip install -q -e .
cd ..

# Download IndicTrans2 model (en-indic direction)
echo "   Downloading IndicTrans2 en-indic model..."
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='ai4bharat/indictrans2-en-indic-1B',
    local_dir='models/indictrans2-en-indic',
    ignore_patterns=['*.msgpack','*.h5','flax_model*']
)
print('   IndicTrans2 model downloaded.')
"

echo "   IndicTrans2 ready."

# ── 3. NLLB (for Hindi translation) ──────────────────────────────────────────
echo ""
echo "[3/5] Downloading NLLB-200-distilled-600M for Hindi translation..."
python3 -c "
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
print('  Downloading NLLB tokenizer...')
AutoTokenizer.from_pretrained('facebook/nllb-200-distilled-600M', cache_dir='models/nllb')
print('  Downloading NLLB model...')
AutoModelForSeq2SeqLM.from_pretrained('facebook/nllb-200-distilled-600M', cache_dir='models/nllb')
print('  NLLB ready.')
"

# ── 4. HuggingFace login for gated models (LLaMA-2) ──────────────────────────
echo ""
echo "[4/5] HuggingFace authentication..."
echo "   NOTE: LLaMA-2-7B requires HuggingFace login."
echo "   Run the following command and enter your HF token:"
echo "     huggingface-cli login"
echo "   Or set environment variable:  export HF_TOKEN=your_token_here"
echo "   (You can skip this if you already have a cached token)"

# ── 5. Verify GPU ─────────────────────────────────────────────────────────────
echo ""
echo "[5/5] Verifying GPU..."
python3 -c "
import torch
if torch.cuda.is_available():
    name = torch.cuda.get_device_name(0)
    mem  = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f'   GPU: {name}')
    print(f'   VRAM: {mem:.1f} GB')
else:
    print('   WARNING: No CUDA GPU detected. Pipeline requires GPU.')
"

echo ""
echo "============================================================"
echo " Setup complete. You can now run:  jupyter notebook"
echo " Then open:  pipeline_notebook.ipynb"
echo "============================================================"
