# The Ghost in the Machine
### Authorship and the Era-Detector Illusion — PreCog NLP Track (2026)

An investigation into whether an AI-text detector learns *authorship*, or just learns
19th-century literary register and treats that as "human". Built across five tasks, each
in its own notebook. The full write-up of the approach, setbacks, and findings is in the
report.

* The code is organised into one notebook per task, in run order (Task 0 → Task 4).
* The approach, design decisions, and every dead-end are written up in the report.
* The `models/` folder contains the trained LoRA adapters, named as referenced in the report (this folder can be viewed when run locally)
  (`tierC_roberta_tertiary` is the canonical model used for Tasks 3 and 4).
* The `data/` folder holds the frozen dataset, the generation log, and the two held-out probe sets.
* The report can be read [here](Ghost_in_the_Machine_Report.pdf).

## Notebooks

| Notebook | What it does |
|----------|--------------|
| `task0_dataset.ipynb` | Builds the 3 genre-matched classes, frozen split, and probe sets |
| `task1_fingerprint.ipynb` | Stylometry: sentence variance, punctuation, hapax, PCA |
| `task2_tiers_ab.ipynb` | Tier A (XGBoost) + Tier B (GloVe FFNN) + leakage audit |
| `task2_tier_c.ipynb` | Tier C (DistilBERT + RoBERTa, both with LoRA) + probe comparison |
| `task3_smoking_gun.ipynb` | Saliency, proper-noun ablation, SOP disentangling, genre probe |
| `task4_turing_ga.ipynb` | Adversarial genetic algorithm + the personal SOP test |

Run them in order. Every tier reads the same frozen split from `data/dataset/`, so Task 0
must run first. Tasks 2C, 3, and 4 load the saved adapter from `models/`.

## Running it

* Developed and tested on **Python 3.11**, in a local `venv`. No Colab.
* Tasks 2C, 3, and 4 train/load transformers and want a GPU; the rest run fine on CPU.
* spaCy needs its model: `python -m spacy download en_core_web_sm`
* Text generation (Task 0) and the GA mutator (Task 4) call the **Groq API**
  (`llama-3.3-70b-versatile`). Set `GROQ_API_KEY` in your environment before running those.
  Gemini was tried first but hit hard rate caps; the switch is explained in the report.

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
export GROQ_API_KEY=your_key_here
```

## Requirements

Minimum libraries:

* `torch`
* `transformers`
* `peft`
* `datasets`
* `scikit-learn`
* `xgboost`
* `gensim`
* `spacy`
* `captum`
* `textstat`
* `numpy`
* `pandas`
* `matplotlib`
* `seaborn`
* `groq`

## Notes

* One stratified split (882 / 189 / 189) is frozen to disk and reused by every notebook, so
  cross-tier comparisons are valid. Seeds fixed at 42.
* Every generation call is logged to `data/dataset/generation_log.jsonl` for auditability.
* The two probe sets (`data/probe_modern_human.jsonl`, `data/probe_sop.jsonl`) are predicted
  but never trained on. They are how the report measures register leakage.