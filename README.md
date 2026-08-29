# LMA Individual Project — Monolingual Transformer Language Models

Two independent decoder-only Transformer LMs (~25M params each), built from scratch.

| | Model H | Model L |
|---|---|---|
| **Language** | Hindi (हिन्दी) | Sindhi (سنڌي) |
| **Resource tier** | Higher-resource | Lower-resource |
| **Corpus (train split)** | 504.8M tokens | 618.4M tokens |
| **Tokenizer** | 16K Unigram (SentencePiece) | 16K Unigram (SentencePiece) |

The two models share no data, vocabulary, or weights — separate corpora, separate
tokenizers, separate training runs.

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

## Google Drive Links

All large artifacts (Datasets, Tokenizer Models) that are too large for Git are securely hosted on Google Drive. Permissions are set to public ("Anyone with the link can view").

### Phase 1 Artifacts
- **[Raw Datasets (Scrapes & Dumps)](https://drive.google.com/drive/folders/1zvLgcQdYhoL8NmwMuf3zTX8XX7RFzjyp?usp=sharing)**
- **[Preprocessed & Cleaned Datasets (Train/Val/Test splits)](https://drive.google.com/drive/folders/1GfBSP6Cbyn7ONmLlZrhtbTkPIhgQczQY?usp=sharing)**
- **[Trained Tokenizer Models & Vocab (16K Unigram)](https://drive.google.com/drive/folders/1S0uDU-52hUb54DrZDFUTji7uX_fk_kFb?usp=sharing)** — also committed directly to this repo under `hindi/tokenizer/` and `sindhi/tokenizer/`, so no download is needed to reproduce Phase 2.

### Reports
- **[Phase 1 report](report/phase1.md)** — dataset statistics, provenance, tokenizer evaluation
- Per-language detail: [`hindi/data/dataset_statistics.md`](hindi/data/dataset_statistics.md),
  [`sindhi/data/dataset_statistics.md`](sindhi/data/dataset_statistics.md),
  [`hindi/tokenizer/tokenizer_statistics.md`](hindi/tokenizer/tokenizer_statistics.md),
  [`sindhi/tokenizer/tokenizer_statistics.md`](sindhi/tokenizer/tokenizer_statistics.md)

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
Trains the 16K Unigram and BPE tokenizers and evaluates them on held-out text.
```bash
bash run_tokenizer_hindi.sh
bash run_tokenizer_sindhi.sh
```

**5. Verify the train/val/test splits**
The split definition lives in `common/preprocessing/make_splits.py` (nominal 95 / 2.5 / 2.5,
seed 42) and is imported by `pipeline.py`, so the two cannot drift apart. To check the
splits on disk against it:
```bash
python -m common.preprocessing.make_splits --language hindi  --verify
python -m common.preprocessing.make_splits --language sindhi --verify
```

**6. Regenerate the statistics reports**
Both `tokenizer_statistics.md` files and `split_statistics.json` are generated from
measured artefacts rather than written by hand, so they can be rebuilt at any time:
```bash
for lang in hindi sindhi; do
  python -m common.tokenizer.evaluate_tokenizer  --language $lang --vocab_size 16000
  python -m common.tokenizer.write_stats_report  --language $lang --vocab_size 16000
  python -m common.preprocessing.split_stats     --language $lang --train_stride 1
done
```

> **Note on the collection scripts.** `run_all_hindi.sh` and `run_all_sindhi.sh` cover the
> bulk public-corpus downloads and the preprocessing/dedup pipeline. The manually-collected
> portion is gathered separately and must be run first if you are rebuilding the corpus
> from nothing:
> ```bash
> python -m common.collection.scrape_wiki_dump --language hindi   # and sindhi
> python run_scrapers.py 300
> ```
