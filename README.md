[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/Q6gOCxoh)

# LMA Individual Project — Monolingual Transformer Language Models

Two independent decoder-only Transformer LMs (~25M params each), built from scratch.

| | Model H | Model L |
|---|---|---|
| **Language** | Hindi (हिन्दी) | Sindhi (سنڌي) |
| **Resource tier** | Higher-resource | Lower-resource |
| **Target tokens** | ~500M | ~500M (max feasible) |

## Repository Structure

```
├── hindi/                      # Model H — Hindi
│   ├── data/
│   │   ├── collection/         # scraping, OCR, manual collection scripts
│   │   ├── preprocessing/      # cleaning, dedup, unicode normalization
│   │   └── splits/             # train / val / test split logic & stats
│   ├── tokenizer/              # BPE/SentencePiece training code & vocab files
│   ├── model/                  # Transformer implementation (MHA, pos emb, causal mask)
│   ├── train/                  # pretraining scripts & checkpoint management
│   ├── finetune/               # reasoning finetuning (Phase 3)
│   ├── eval/                   # evaluation (PPL, BPB, BLEU, chrF, ROUGE-L, diversity)
│   ├── attention/              # attention heatmaps, entropy, mean distance analysis
│   └── configs/                # model & training hyperparameter configs
│
├── sindhi/                     # Model L — Sindhi (same structure as hindi/)
│   ├── data/
│   │   ├── collection/
│   │   ├── preprocessing/
│   │   └── splits/
│   ├── tokenizer/
│   ├── model/
│   ├── train/
│   ├── finetune/
│   ├── eval/
│   ├── attention/
│   └── configs/
│
├── utils/                      # shared utilities (plotting, logging, etc.)
├── report/                     # per-phase reports, plots, tables, heatmaps
└── README.md
```

## Phases & Deadlines

| # | Phase | Deadline | Marks |
|---|---|---|---|
| 1 | Data Collection & Tokenizer Construction | 19 Aug 2026 | 25 |
| 2 | Model Implementation, Pretraining & Evaluation | 5 Sep 2026 | 40 |
| 3 | Reasoning Finetuning, Attention Analysis & Final Report | 16 Sep 2026 | 35 |

## Google Drive Links

> **Checkpoints & large artifacts will be linked here once available.**

- [ ] Hindi pretrained checkpoint
- [ ] Sindhi pretrained checkpoint
- [ ] Hindi finetuned checkpoint
- [ ] Sindhi finetuned checkpoint
- [ ] Datasets (if too large for git)

## Reproduction

### Phase 1: Data Collection & Tokenizer Construction

**1. Setup Environment**
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**2. Run Hindi Pipeline**
```bash
# 1. Download/scrape raw data (~1 hour)
python -m hindi.data.collection.download_public_data
python -m common.collection.scrape_wiki_dump --language hindi

# 2. Run streaming cleaning, deduplication, and 95/2.5/2.5 train/val/test splitting
python -m common.preprocessing.pipeline --language hindi

# 3. Train Unigram & BPE tokenizers
bash run_tokenizer_hindi.sh
```

**3. Run Sindhi Pipeline**
```bash
# 1. Download/scrape raw data (~1 hour)
python -m sindhi.data.collection.download_public_data
python -m common.collection.scrape_wiki_dump --language sindhi
python sindhi/data/collection/extract_arnizamani.py

# 2. Run streaming cleaning, deduplication, and 95/2.5/2.5 train/val/test splitting
python -m common.preprocessing.pipeline --language sindhi

# 3. Train Unigram & BPE tokenizers
bash run_tokenizer_sindhi.sh
```

**4. Evaluate Tokenizers**
```bash
python -m common.tokenizer.evaluate_tokenizer --language hindi
python -m common.tokenizer.evaluate_tokenizer --language sindhi
```
