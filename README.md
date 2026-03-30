# LLM Bias Evaluation Pipeline — Category-Level Analysis
## Mohanraj Ramanujam | PA2312049010014 | M.Tech AI, SRM University

---

## Pipeline Structure

```
pipeline/
├── setup.sh                    ← Run ONCE on fresh RunPod to install dependencies
├── make_notebook.py            ← Converts .py script to .ipynb Jupyter notebook
├── pipeline_notebook.py        ← Main pipeline (open as notebook after conversion)
├── data/
│   ├── stereo_type_dataset.csv ← YOUR INPUT: stereotype pairs
│   └── toxicity.csv            ← YOUR INPUT: toxicity prompts
├── modules/
│   ├── config.py               ← All settings (paths, model configs, batch sizes)
│   ├── validate_data.py        ← Step 1: Validate CSV columns
│   ├── translate.py            ← Step 2: Translate to Tamil (IndicTrans2) + Hindi (NLLB)
│   ├── evaluate_stereotype.py  ← Step 3: Log-likelihood scoring per category
│   ├── evaluate_toxicity.py    ← Step 4: Text generation + Detoxify subtype scoring
│   └── visualize.py            ← Step 5: Generate all charts
├── outputs/                    ← All results saved here
├── checkpoints/                ← Per-model-per-language checkpoint files
└── logs/                       ← Log files
```

---

## Required CSV Format

### stereo_type_dataset.csv
| Column     | Description                          | Example                          |
|------------|--------------------------------------|----------------------------------|
| sent_more  | Stereotypical sentence               | "Women are bad at math."         |
| sent_less  | Anti-stereotypical sentence          | "Men are bad at math."           |
| bias_type  | Category label                       | gender, religion, race-color...  |

### toxicity.csv
| Column | Description              | Example                           |
|--------|--------------------------|-----------------------------------|
| prompt | Text prompt for generation| "The politician announced that..." |

**If your column names are different**, update `modules/config.py`:
```python
STEREO_COL_MORE = "sent_more"     # change to your column name
STEREO_COL_LESS = "sent_less"
STEREO_COL_CAT  = "bias_type"
TOXICITY_COL_PROMPT = "prompt"
```

---

## Step-by-Step Usage on RunPod RTX 4090

### 1. First-time setup (run ONCE)
```bash
bash setup.sh
```

### 2. Place your data files
```bash
cp /path/to/stereo_type_dataset.csv  data/
cp /path/to/toxicity.csv             data/
```

### 3. Set HuggingFace token (for LLaMA-2)
```bash
export HF_TOKEN=your_token_here
huggingface-cli login
```

### 4. Convert to Jupyter notebook and open
```bash
python3 make_notebook.py
jupyter notebook pipeline_notebook.ipynb
```

### 5. Run cells in order (1 → 8)
Each cell saves checkpoints. If the run crashes, re-run from Cell 1 —
completed model×language combinations will be detected and skipped.

---

## Outputs

| File | Description |
|------|-------------|
| `outputs/stereo_English/Tamil/Hindi.csv` | Translated stereotype datasets |
| `outputs/toxicity_English/Tamil/Hindi.csv` | Translated toxicity datasets |
| `outputs/stereo_raw_predictions.csv` | All pair-level log-likelihood results |
| `outputs/toxicity_raw_scored.csv` | All continuation-level Detoxify scores |
| `outputs/stereotype_scores.csv` | Overall SBS + CSBS per category per model per language |
| `outputs/toxicity_scores.csv` | STBS per subtype per model per language |
| `outputs/stereotype_scores.json` | Same as CSV in JSON format |
| `outputs/toxicity_scores.json` | Same as CSV in JSON format |
| `outputs/charts/fig1_sbs_english.png` | Category-level SBS chart — English |
| `outputs/charts/fig2_sbs_tamil.png` | Category-level SBS chart — Tamil |
| `outputs/charts/fig3_sbs_hindi.png` | Category-level SBS chart — Hindi |
| `outputs/charts/fig4_tbs_english.png` | Subtype-level TBS chart — English |
| `outputs/charts/fig5_tbs_tamil.png` | Subtype-level TBS chart — Tamil |
| `outputs/charts/fig6_tbs_hindi.png` | Subtype-level TBS chart — Hindi |
| `outputs/charts/fig7_sbs_crosslingual.png` | Cross-lingual SBS comparison |
| `outputs/charts/fig8_tbs_crosslingual.png` | Cross-lingual TBS comparison |

---

## Estimated Runtimes on RTX 4090

| Step | Estimated Time |
|------|----------------|
| Translation (Tamil + Hindi, both datasets) | 45–90 min |
| LLaMA-2-7B stereotype (3 languages) | ~60 min |
| BLOOM-560M stereotype (3 languages) | ~25 min |
| Falcon-1B stereotype (3 languages) | ~30 min |
| Mistral-7B stereotype (3 languages) | ~60 min |
| GPT-J-6B stereotype (3 languages) | ~50 min |
| All 5 models — toxicity generation + scoring | ~90 min |
| Visualisation | ~2 min |
| **Total** | **~6–8 hours** |

---

## Troubleshooting

**"IndicTransToolkit not found"**
→ Run `bash setup.sh` again. IndicTrans2 requires cloning from GitHub.

**"CUDA out of memory"**
→ Reduce batch sizes in `config.py`: `stereo_batch` and `toxicity_batch`.
→ GPT-J-6B already uses batch=8; try reducing to 4 if OOM occurs.

**"NaN loss for BLOOM-560M"**
→ Already handled: BLOOM-560M is forced to float32 in config.py.

**"Model requires authentication" (LLaMA-2)**
→ Run `huggingface-cli login` and enter your HF token.
→ Or: `export HF_TOKEN=hf_xxx` before running the notebook.

**"Column not found in CSV"**
→ Run Cell 2 (validate_data). It will print exact column names found.
→ Update `config.py` with your actual column names.

**Pipeline crashed mid-run**
→ Re-run from Cell 1. Completed model×language checkpoints are detected automatically.
→ Each checkpoint is stored in `checkpoints/` as a separate CSV.
