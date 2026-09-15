# LMA Individual Project — Monolingual Transformer Language Models

Two independent decoder-only Transformer LMs (~25M params each), built from scratch.

| | Model H | Model L |
|---|---|---|
| **Language** | Hindi (हिन्दी) | Sindhi (سنڌي) |
| **Resource tier** | Higher-resource | Lower-resource |
| **Corpus (train split)** | 504.8M tokens | 618.4M tokens |
| **Tokenizer** | 16K Unigram (SentencePiece) | 16K Unigram (SentencePiece) |
| **Parameters** | 24,220,672 | 24,482,816 |
| **Context length** | 512 | 1024 |
| **Trained on** | 500,023,296 tokens | 500,023,296 tokens |
| **Validation perplexity** | 32.44 | 51.97 |
| **Bits per byte** (test, matched ctx) | **0.5375** | **0.9584** |
| **Reasoning accuracy, pretrained** | 0.0% | 0.0% |
| **Reasoning accuracy, finetuned** (in-distribution; chance ≈37%) | **78.8%** | **73.7%** |
| **Bits per byte after finetuning** (test, matched ctx) | 1.0111 | 1.9563 |

The two models share no data, vocabulary, or weights — separate corpora, separate
tokenizers, separate training runs.

**Final report: [`report/final_report.pdf`](report/final_report.pdf)** — a single LaTeX
report covering all three phases.

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
- **[Final report (PDF, all three phases)](report/final_report.pdf)**
- **Phase 3 supporting material**
  - [All Phase 3 tables](report/phase3_results.md) — generated from the results files
  - [Reasoning task design](report/phase3_task_design.md) — dataset specification, with dated amendments
  - [Lexicon validation](report/phase3_lexicon_validation.md) — corpus evidence for every word the dataset uses
- **[Phase 2 report](report/phase2.md)** — architecture, pretraining, evaluation, resource comparison
  - [Intrinsic evaluation](report/phase2_intrinsic.md) — perplexity, bits-per-byte
  - [Generation quality](report/phase2_generation.md) — BLEU, chrF++, ROUGE-L, samples
  - [Attention analysis](report/phase2_attention.md) — heatmaps, entropy, distance
  - [Context length decision](report/context_length_decision.md)
  - Figures: [`report/figures/`](report/figures/)
- **[Phase 1 report](report/phase1.md)** — dataset statistics, provenance, tokenizer evaluation
- Per-language detail: [`hindi/data/dataset_statistics.md`](hindi/data/dataset_statistics.md),
  [`sindhi/data/dataset_statistics.md`](sindhi/data/dataset_statistics.md),
  [`hindi/tokenizer/tokenizer_statistics.md`](hindi/tokenizer/tokenizer_statistics.md),
  [`sindhi/tokenizer/tokenizer_statistics.md`](sindhi/tokenizer/tokenizer_statistics.md)

### Phase 2 Artifacts — pretrained checkpoints

- **Model H (Hindi) pretrained checkpoint** — `https://drive.google.com/drive/folders/10fy4e-8qppn0Xd9gEAXP00pZov9xOGzR?usp=sharing`
- **Model L (Sindhi) pretrained checkpoint** — `https://drive.google.com/drive/folders/10fy4e-8qppn0Xd9gEAXP00pZov9xOGzR?usp=sharing`


Each checkpoint is ~280 MB and contains model weights, optimizer state, scheduler position,
step count and the model config — everything needed to resume or evaluate.

- **Tokenized corpora** (`<language>/data/tokenized/{train,val,test}.bin`, ~2.3 GB) — `https://drive.google.com/drive/folders/1siTlcWp43a7G8MnHZK8RTuwOxP_Meq7z?usp=sharing` — also reproducible from the preprocessed datasets with `python -m common.data.tokenize_corpus --language <language>`

### Phase 2 artifacts in this repository

The brief requires training logs and loss curves to be *in the branch*, not on an external
dashboard. They are:

| Artifact | Location |
|---|---|
| Training log — Model H | [`hindi/train/run/train.log`](hindi/train/run/train.log) |
| Training log — Model L | [`sindhi/train/run/train.log`](sindhi/train/run/train.log) |
| Loss curves, raw (per-step JSONL) | [`hindi/train/run/training_log.jsonl`](hindi/train/run/training_log.jsonl), [`sindhi/train/run/training_log.jsonl`](sindhi/train/run/training_log.jsonl) |
| Loss curves, plotted | [`report/figures/training_curves.png`](report/figures/training_curves.png) |
| Model configs saved beside checkpoints | `hindi/train/run/model_config.json`, `sindhi/train/run/model_config.json` |
| Attention heatmaps (early + late layer, both models) | [`report/figures/`](report/figures/) — `attention_*.png` |
| Intrinsic / generation / attention measurements | `report/intrinsic_results.json`, `generation_results.json`, `attention_results.json` |

Checkpoints themselves are on Drive (above) because they are ~280 MB each; everything else
above is committed.

### Phase 3 Artifacts — finetuned checkpoints and reasoning datasets

- **Model H (Hindi) finetuned checkpoint** — `https://drive.google.com/file/d/1Vn6avgCe3TGpX7B73qWK_8jbGCxhOxGz/view?usp=sharing` — `hindi_best.pt` (step 3600, the checkpoint behind every reported result)
- **Model L (Sindhi) finetuned checkpoint** — `https://drive.google.com/file/d/1LZ55dCUhvY2MS70naQWHTORaJXHzN_yh/view?usp=sharing` — `sindhi_best.pt` (step 3750, the checkpoint behind every reported result; also the final, resumable state)
- **Reasoning datasets** (train / val / four test splits, both languages) — `https://drive.google.com/drive/folders/1unE3WnGcQMbEhzrxkLkh6WgP-MV1G7VM?usp=sharing` — also fully reproducible with `python -m common.reasoning.generate`

Same resume-capable format as the pretrained checkpoints.

### Phase 3 artifacts in this repository

| Artifact | Location |
|---|---|
| Dataset generation | `common/reasoning/` — `generate.py`, `templates_{hindi,sindhi}.json`, `lexicon_{hindi,sindhi}.json`, `validate_lexicon.py` |
| Dataset statistics and leakage audit | `<language>/finetune/data/dataset_statistics.json` |
| Finetuning code | `common/finetune/` — `dataset.py`, `trainer.py`, `run.py`, `lr_probe.py` |
| Finetuning hyperparameters | `<language>/finetune/run/train_config.json`, `model_config.json` |
| Finetuning logs | `<language>/finetune/run/finetune.log`, `training_log.jsonl`; learning-rate probe in `<language>/finetune/lr_probe/` |
| Finetuning curves | `report/figures/finetune_curves.png`, `finetune_lr_probe.png` |
| Reasoning results (pretrained 0/4-shot, finetuned, forced choice) | `report/reasoning_*.json`; every prediction in `<language>/eval/*.jsonl` |
| Error analysis | `report/reasoning_analysis.json`, `report/phase3_results.md` |
| Pretrained vs finetuned attention | `report/attention_finetune_results.json`, `report/figures/attention_finetune_*.png` |
| Language modelling after finetuning | `report/intrinsic_results_finetuned*.json`, `report/figures/finetune_lm_regression.png` |
| Full-sequence-loss ablation | `scripts/run_fullseq_ablation.sh`; `<language>/finetune/run_fullseq/`; `report/*fullseq*` |

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

### Phase 2: Model Implementation, Pretraining & Evaluation

```bash
# 1. Tokenize the corpora into uint16 .bin files
python -m common.data.tokenize_corpus --language hindi
python -m common.data.tokenize_corpus --language sindhi

# 2. Generate the per-language model configs
python -m common.model.make_configs

# 3. Pretrain. Safe to re-run: resumes from the last checkpoint after any interruption,
#    refuses to start on battery power, and inhibits sleep for the duration.
bash scripts/train_local.sh hindi
bash scripts/train_local.sh sindhi

# 4. Evaluate
python -m common.eval.intrinsic
python -m common.eval.generation --split val --n_prompts 200
python -m common.eval.attention --split val

# 5. Figures and reports
python -m common.eval.plot_training
python -m common.eval.plot_attention
python -m common.eval.write_intrinsic_report
python -m common.eval.write_generation_report
python -m common.eval.write_attention_report
```

To train on Kaggle instead (needed if your GPU has under ~6 GB), see `kaggle/` — build the
upload archives with `package_tokens_upload.sh` and `package_code_upload.sh`, then run
`verify_kaggle_setup.py` followed by `train_kaggle.py` as two notebook cells.

### Phase 3: Reasoning Finetuning, Attention Analysis & Final Report

Requires the Phase 2 pretrained checkpoints at `<language>/train/run/best.pt` (Drive links
above).

```bash
# 1. Build the reasoning datasets (seeded; ~5 seconds). Optional checks first:
python -m common.reasoning.validate_lexicon --scan_train   # re-derive the corpus-validated lexicons
python -m common.reasoning.check_templates                 # validate the templates
python -m common.reasoning.generate

# 2. The whole finetuning experiment, unattended (~40 min on an RTX 3050): learning-rate
#    probe, finetuning both models, greedy evaluation, 4-shot baseline. Idempotent and
#    resumable. systemd-run keeps it alive if the launching terminal or editor closes.
systemd-run --user --unit=lma-finetune --collect --working-directory="$PWD" \
    bash scripts/run_finetuning.sh
#    ...or one model at a time:  bash scripts/finetune_local.sh hindi

# 3. Baselines, forced choice and error analysis
python -m common.finetune.evaluate --checkpoint pretrained --shots 0
python -m common.finetune.rank_candidates --checkpoint finetuned
python -m common.finetune.rank_candidates --checkpoint pretrained
python -m common.finetune.analyse_reasoning

# 4. Attention before and after finetuning; language modelling after finetuning
python -m common.finetune.attention_compare
python -m common.eval.intrinsic --all --checkpoint finetuned --out report/intrinsic_results_finetuned.json

# 5. Ablation: loss on the full sequence (~30 min)
systemd-run --user --unit=lma-fullseq --collect --working-directory="$PWD" \
    bash scripts/run_fullseq_ablation.sh

# 6. Figures and the generated tables
python -m common.finetune.plot_reasoning
python -m common.finetune.plot_attention_compare
python -m common.finetune.write_results_report
```

**Tests.** 142 across model, data, training, evaluation and finetuning, all passing:
```bash
python -m common.model.tests.test_embeddings     # and test_attention, test_block, test_gpt
python -m common.data.tests.test_data
python -m common.train.tests.test_trainer
python -m common.eval.tests.test_intrinsic       # and test_generation, test_attention
python -m common.finetune.tests.test_dataset
```

> **Note on the collection scripts.** `run_all_hindi.sh` and `run_all_sindhi.sh` cover the
> bulk public-corpus downloads and the preprocessing/dedup pipeline. The manually-collected
> portion is gathered separately and must be run first if you are rebuilding the corpus
> from nothing:
> ```bash
> python -m common.collection.scrape_wiki_dump --language hindi   # and sindhi
> python run_scrapers.py 300
> ```
