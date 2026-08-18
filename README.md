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

All large artifacts (Datasets, Tokenizer Models) that are too large for Git are securely hosted on Google Drive. Permissions are set to public ("Anyone with the link can view").

### Phase 1 Artifacts
- **[Raw Datasets (Scrapes & Dumps)](https://drive.google.com/drive/folders/1zvLgcQdYhoL8NmwMuf3zTX8XX7RFzjyp?usp=sharing)**
- **[Preprocessed & Cleaned Datasets (Train/Val/Test splits)](https://drive.google.com/drive/folders/1GfBSP6Cbyn7ONmLlZrhtbTkPIhgQczQY?usp=sharing)**
- **[Trained Tokenizer Models & Vocab (16K Unigram)](https://drive.google.com/drive/folders/1S0uDU-52hUb54DrZDFUTji7uX_fk_kFb?usp=sharing)**

### Phase 2 & 3 Artifacts (Pending)
- [ ] Hindi pretrained checkpoint
- [ ] Sindhi pretrained checkpoint
- [ ] Hindi finetuned checkpoint
- [ ] Sindhi finetuned checkpoint

## Reproduction

### Phase 1: Data Collection & Tokenizer Construction

If you do not wish to download the pre-compiled datasets from the Google Drive links above, you can reproduce the entire pipeline from absolute scratch.

**1. Setup Environment**
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**2. Run the Full Hindi Pipeline**
This master script automatically handles scraping, heuristic filtering, exact deduplication, retrospective MinHash LSH fuzzy deduplication, and final token counting.
```bash
bash run_all_hindi.sh
```

**3. Run the Full Sindhi Pipeline**
This master script handles the entire Sindhi ingestion process from raw scrapes to final token counts.
```bash
bash run_all_sindhi.sh
```

**4. Train and Evaluate Tokenizers**
Once the data is preprocessed, train the 16K Unigram/BPE tokenizers and evaluate their fertility.
```bash
bash run_tokenizer_hindi.sh
bash run_tokenizer_sindhi.sh
```
