# Phase 2 — Model Implementation, Pretraining and Evaluation

Two independent decoder-only Transformer language models, built from scratch in PyTorch,
pretrained on ~500M tokens each and evaluated on their own held-out data.

Builds directly on Phase 1: each model uses its own corpus and its own 16K Unigram
tokenizer from that phase, with no data, vocabulary or weights shared between the two.
See [`phase1.md`](phase1.md) for corpus construction and tokenizer selection, including a
revision log of corrections made to that report after its deadline.

**Companion documents:** [`phase2_intrinsic.md`](phase2_intrinsic.md) (perplexity, BPB) ·
[`phase2_generation.md`](phase2_generation.md) (BLEU, chrF++, ROUGE-L, samples) ·
[`phase2_attention.md`](phase2_attention.md) (heatmaps, entropy, distance) ·
[`context_length_decision.md`](context_length_decision.md)

| | Model H | Model L |
|---|---|---|
| Language | Hindi (Devanagari) | Sindhi (Perso-Arabic) |
| Resource tier | Higher | Lower |
| Parameters | **24,220,672** | **24,482,816** |
| Context length | 512 | 1024 |
| Training tokens | 500,023,296 | 500,023,296 |
| Best validation loss | **3.4795** | **3.9506** |
| Validation perplexity | **32.44** | **51.97** |
| **Bits per byte (test, matched 512 ctx)** | **0.5375** | **0.9584** |

**Headline result: Model L needs 78.3% more bits per byte than Model H.**

---

## 1. Architecture

Implemented from primitive PyTorch layers only — `nn.Linear`, `nn.Embedding`,
`nn.LayerNorm`, `nn.Dropout`. No `nn.Transformer*`, no HuggingFace model classes, no
pre-built attention. Attention is written out as explicit tensor operations, which is also
what makes the Section 5 analysis possible: a fused kernel never materialises the
probability matrix.

| Setting | Value | Justification |
|---|---|---|
| `d_model` | 512 | see depth/width trade-off below |
| `n_layers` | 6 | " |
| `n_heads` | 8 | gives `d_head` = 64, a standard per-head width |
| `d_ff` | 1536 (3×) | the knob used to hit the ~25M budget |
| `max_seq_len` | 512 (H) / 1024 (L) | matched to each corpus; see §2 |
| `dropout` | 0.1 | on embeddings, attention probabilities, sublayer outputs |
| `tie_embeddings` | true | saves 8.19M parameters |

### Parameter budget

| Component | Model H | Model L |
|---|---|---|
| Token embedding (16,000 × 512) | 8,192,000 | 8,192,000 |
| Position embedding | 262,144 | 524,288 |
| 6 transformer blocks | 15,765,504 | 15,765,504 |
| Final LayerNorm | 1,024 | 1,024 |
| LM head | 0 (tied) | 0 (tied) |
| **Total** | **24,220,672** | **24,482,816** |

The models differ by exactly **262,144 parameters** — precisely the extra position-table
rows Model L needs. **The transformer stacks are byte-for-byte identical at 15,765,504
parameters each**, which is asserted by `make_configs.py --check` and is what allows the
comparison in §7 to attribute differences to language and data rather than to capacity.

### Depth/width trade-off

Three configurations land within 0.5% of 24.2M: (d=512, L=6, ff=1536), (d=448, L=7,
ff=1792) and (d=512, L=5, ff=2048). We chose the first because depth buys more than
feed-forward width at this scale, and `d_model=512` with 8 heads keeps `d_head` at 64 with
power-of-two dimensions. The cost is a 3× feed-forward ratio rather than the conventional
4×: that ratio is the knob we gave up to protect depth and head dimension, both of which
matter more.

### Weight tying

The output projection reuses the token embedding matrix — the same tensor, not a copy.
This saves **16,000 × 512 = 8,192,000 parameters, roughly a quarter of the untied model**.
At a 25M budget that is the difference between a 6-layer stack and roughly a 3-layer one.
Verified as genuine sharing: vocabulary rows never seen as *input* still receive gradient,
which is only possible if the output projection uses the same matrix.

### Causal masking, verified empirically

The brief asks for a demonstration that the model cannot see the future. For every position
*t*, the test perturbs **all** tokens after *t* and asserts that the logits at positions
0..*t* are **bitwise identical** (`torch.equal`, not `allclose`), then asserts that position
*t+1* **did** change so the test cannot pass vacuously.

The test was validated by sabotage: with the causal mask disabled, it fails immediately, as
do the checks that attention weights are lower-triangular and that position 0 attends only
to itself. Measured attention mass above the diagonal is **exactly 0.0**, not merely small.

---

## 2. Context length: the one asymmetry

Model L uses a 1024-token context where Model H uses 512. Measured document lengths:

| | median | mean | fits in 512 | fits in 1024 |
|---|---|---|---|---|
| Hindi | 64 tokens | 85 | 99.3% | 99.8% |
| Sindhi | 219 tokens | 774 | 74.5% | 86.4% |

Hindi documents are short news and encyclopaedia paragraphs — a 512-token window already
spans roughly eight of them, and past an `<|endoftext|>` boundary the earlier text is a
different article, so extra context adds noise rather than signal. Sindhi documents are
long-form books and articles at 3.4× the median length, so a 512-token window frequently
sits *inside* a single document with the earlier half of the same passage cut off.

This is not a tokenizer artifact: fertility is nearly identical (1.3056 vs 1.2987
tokens/word), so the gap reflects genuinely longer documents.

**Threat to validity, and how it is handled.** Longer context lowers loss for free. Model L
is therefore evaluated **twice** — at its native 1024 and again at 512 — and every
cross-model claim uses the matched-context figures. Full reasoning in
`context_length_decision.md`.

---

## 3. Pretraining

Identical optimisation for both models:

| Setting | Value |
|---|---|
| Optimizer | AdamW, betas (0.9, 0.95) |
| Peak / floor learning rate | 6e-4 → 6e-5 |
| Schedule | 500-step linear warmup, then cosine decay |
| Weight decay | 0.1, applied to matrices only |
| Gradient clipping | 1.0 |
| Effective batch | **24,576 tokens/step** (both models) |
| Optimizer steps | **20,346** (both models) |
| Precision | bf16 (local) / fp16 + GradScaler (Kaggle T4) |
| Seed | 1337 |

Micro-batch differs (H: 8×6×512, L: 4×6×1024) only because of what fits in 4 GB of VRAM at
each context length. Gradient accumulation makes the optimizer see the same batch either
way — **the product is identical at 24,576 tokens**, which is what matters for the
comparison.

Weight decay is applied only to tensors with ≥2 dimensions. Biases and LayerNorm gains have
one parameter per channel rather than per connection, so decaying them biases the activation
statistics rather than acting as a capacity penalty.

**Figure:** `figures/training_curves.png` — loss, perplexity, learning rate and gradient
norm against tokens consumed.

### Token budget, not epochs

Both models consumed **500,023,296 tokens** = 0.99 epoch-equivalents of the Hindi corpus and
0.81 of the Sindhi. Because sequences are drawn from uniformly random offsets rather than
walked in order, unique coverage is lower than those ratios suggest: **62.9%** of the Hindi
corpus and **55.4%** of the Sindhi were seen at least once, covered regions ~1.5 times on
average. No data was deliberately repeated.

Matching the budget is deliberate. Hindi had 504.8M tokens available and Sindhi 618.4M; 500M
is the largest budget both can meet without repetition, and it is near compute-optimal
(~20 tokens per parameter ≈ 484M, so 1.03×). Capping Sindhi discarded 118M available tokens
— the price of a controlled comparison.

### Training runs

| | Hardware | Wall clock | Throughput |
|---|---|---|---|
| Model H | Kaggle T4 (fp16) | 11.94 h | ~11,800 tok/s |
| Model L | Local RTX 3050 (bf16) | 7.66 h | ~18,400 tok/s |

The laptop GPU outran the T4 on this workload: our hand-rolled attention leans on
elementwise softmax work that does not use Turing tensor cores well, and the RTX 3050's
Ampere architecture supports bf16 where the T4 does not.

### Checkpointing

Every checkpoint contains model weights, optimizer state, scheduler position, gradient-scaler
state, step counter, tokens consumed, RNG states and the model config. Resume was **verified,
not assumed**: a run was hard-killed with SIGKILL mid-training, resumed, and produced weights
**bitwise identical** (`0.00e+00` difference across all 101 tensors) to an uninterrupted run.

Two bugs were found and fixed by that testing:

1. `torch.load(map_location="cuda")` moved saved RNG states onto the GPU, where
   `Generator.set_state` rejects them. A CPU-only test could not see this.
2. A signal arriving mid gradient-accumulation saved an inconsistent state: the data cursor
   and RNG had advanced but the step counter had not, so a resumed run silently trained on
   different data and drifted by ~1.7e-02. Random state is now snapshotted only at completed
   step boundaries.

---

## 4. Intrinsic evaluation

Full detail: `phase2_intrinsic.md`. Figure: `figures/intrinsic_comparison.png`.

| Model | Split | Context | Loss | Perplexity | BPB |
|---|---|---|---|---|---|
| H | val | 512 | 3.4929 | 32.88 | **0.5056** |
| H | test | 512 | 3.5722 | 35.59 | **0.5375** |
| L | val | 1024 (native) | 3.9541 | 52.15 | **0.8821** |
| L | val | 512 (matched) | 3.9960 | 54.38 | **0.8915** |
| L | test | 1024 (native) | 2.9940 | 19.97 | **0.9503** |
| L | test | 512 (matched) | 3.0194 | 20.48 | **0.9584** |

Every token in each split is scored exactly once using non-overlapping windows, so these are
deterministic values rather than sampled estimates.

**Next-token accuracy** (validation, matched 512 context), for a more intuitive reading:

| | top-1 | top-5 | chance |
|---|---|---|---|
| Model H | **35.85%** | 59.01% | 0.006% |
| Model L | **28.37%** | 49.47% | 0.006% |

Model H guesses the exact next token correctly more than a third of the time — roughly
6,000× better than chance.

### Why bits-per-byte is the headline metric

Perplexity is per *token*, and the two models have independently learned tokenizers, so a
token means a different quantity of text in each. BPB normalises by underlying UTF-8 bytes
and is therefore tokenizer-independent.

This corpus supplies an unusually sharp demonstration. On Sindhi, perplexity claims the test
split is **2.6× easier** than validation (19.97 vs 52.15) — not credible for two halves of
one collection. BPB says the opposite and correct thing: test is slightly *harder* (0.9503
vs 0.8821). Perplexity was misled by tokenization; BPB was not.

### Corpus contamination

Investigating that anomaly revealed **encoding-corrupted text (mojibake)** in both test
splits: legacy non-Unicode fonts decoded as Unicode, producing valid script characters in
meaningless order, typically with digits wedged inside words.

| | val | test |
|---|---|---|
| Hindi | 0.6% | **7.3%** |
| Sindhi | 1.6% | **16.6%** |

The corrupted text is short and highly repetitive, hence easy to predict, which deflates
per-token perplexity wherever it is dense.

**Why the Phase 1 filters missed it:** `script_filter` required ≥90% of alphabetic characters
to fall in the expected Unicode block. Mojibake passes comfortably — it *is* Devanagari or
Arabic-block characters, merely in nonsense sequence. Detecting it needs a different signal,
such as the density of digits appearing inside words.

**Robustness of the headline.** The two test splits are contaminated at different rates, so
some of the gap could in principle be contamination asymmetry. The validation splits are
roughly ten times cleaner and give **+76.3%**, against **+78.3%** on test — two independent
estimates at very different contamination levels agreeing within two percentage points. The
finding is not an artifact of corrupted data.

*Not done:* the contaminated documents were measured and documented but not filtered out,
and results were not recomputed on cleaned splits.

---

## 5. Generation quality

Full detail and samples: `phase2_generation.md`. Figure: `figures/generation_metrics.png`.

200 prompts of 64 tokens per model, continued for 128 tokens, scored against the true
continuation.

| Model | Setting | BLEU | chrF++ | ROUGE-L | Distinct-2 | Repetition |
|---|---|---|---|---|---|---|
| H | greedy | 1.46 | 10.88 | 0.089 | 0.122 | **0.810** |
| H | T=0.5 | **1.78** | 15.92 | **0.115** | 0.381 | 0.352 |
| H | T=1.0 | 1.24 | **19.72** | 0.107 | 0.824 | 0.009 |
| H | T=1.5 | 0.19 | 16.63 | 0.054 | 0.988 | 0.000 |
| L | greedy | 1.60 | 10.93 | 0.085 | 0.106 | **0.826** |
| L | T=0.5 | **1.87** | 13.92 | 0.100 | 0.289 | 0.513 |
| L | T=1.0 | 1.42 | **19.50** | 0.099 | 0.858 | 0.012 |
| L | T=1.5 | 0.34 | 16.49 | 0.045 | 0.994 | 0.000 |

**Greedy decoding degenerates**: over 80% of 4-grams repeat within a single sample. **The
metrics disagree about the best temperature** — BLEU and ROUGE-L peak at 0.5, chrF++ at 1.0
— because BLEU rewards exact word matches while chrF tolerates the morphological variation
that dominates both these languages. **The absolute values are low by nature**: open-ended
continuation has no single correct answer, so BLEU 1–2 is expected and does not indicate a
broken model.

Model H edges Model L at T=1.0 (chrF++ 19.72 vs 19.50), agreeing in direction with the
intrinsic results — but by a far smaller margin than the 78% BPB gap. Overlap metrics are
weak instruments for open-ended generation.

---

## 6. Attention analysis

Full detail: `phase2_attention.md`. Figures: `figures/attention_*.png` (6 files).

Measured on 64 held-out windows of 256 tokens per model. Query positions below 32 are
excluded: under a causal mask, early queries are *forced* to be short-range and low-entropy,
which would confound "this head is local" with "this head had nowhere else to look".

**Both models independently learned the same three-phase structure with depth:**

| Phase | Layers | Entropy (H / L) | Distance (H / L) | Behaviour |
|---|---|---|---|---|
| Broad survey | 0–1 | 0.95 / 0.94 | 67 / 67 | near-uniform mixing |
| Specialisation | 2–4 | 0.48 / 0.52 | 22 / 28 | sharp, short-range, positional |
| Re-integration | 5 | 0.65 / 0.60 | 74 / 78 | broad again, long-range |

**Both also converged on a near-pure previous-token head** — Model H L3H7 (entropy 0.030,
distance 1.06, 94% of mass on *t−1*) and Model L L4H4 (entropy 0.018, distance 1.15, 96%).
This head ignores content entirely and reads purely by offset; it is a known component of
induction circuits. Two separately trained models finding it is evidence it is a genuinely
useful solution rather than one run's accident.

**Attention sink.** 16.9% (H) and 23.8% (L) of all attention mass lands on position 0
regardless of what that token is. Softmax forces every row to sum to 1, so a head with
nothing relevant to attend to must place its mass somewhere, and the first position is a
stable default. Model L relies on it 1.4× more, which *may* indicate more positions where it
finds nothing useful — stated as suggestive, since establishing cause would need a controlled
experiment.

---

## 7. Resource-level comparison

**Model H outperforms Model L by 78.3% in bits per byte at matched context.** The
architectures are identical, the token budgets are identical, the optimisation is identical.
The difference is the languages and their corpora.

### The resource gap is qualitative, not quantitative

The obvious explanation — the low-resource language had less data — **is false here**:

| | corpus (train) | tokens used | fraction used |
|---|---|---|---|
| Hindi | 504,818,280 | 500,023,296 | 99.1% |
| Sindhi | 618,440,091 | 500,023,296 | 80.9% |

**Sindhi had 22% more raw text available.** The gap is in the *composition* of that text:

1. **Source diversity.** Hindi came from broad web-scale corpora (AI4Bharat Sangraha,
   IndicCorp V2) spanning many domains. Sindhi came from a handful of long-form book and
   newspaper repositories — more tokens, drawn from a narrower distribution.

2. **Encoding corruption.** 16.6% of the Sindhi test split is mojibake, against 7.3% for
   Hindi. Corruption that survives filtering is a direct consequence of a language having
   less clean digital text and more legacy-font material.

3. **Document structure.** Sindhi documents are 3.4× longer at the median, which is why the
   two models needed different context lengths.

### Direct evidence: the generalisation gap

The training curves show something the aggregate metrics do not:

| | final train loss | validation loss | gap |
|---|---|---|---|
| Model H | 3.6018 | 3.4795 | **−0.12** |
| Model L | 3.5286 | 3.9506 | **+0.42** |

**Model H shows no generalisation gap at all** — its validation loss is *below* its training
loss, which is normal when training loss is measured on single noisy batches with dropout
active while validation is measured with dropout off.

**Model L shows a substantial +0.42 gap.** It fits its training data slightly *better* than
Model H fits its own (3.53 vs 3.60), yet generalises considerably worse. That is the
signature of a narrower, more repetitive corpus: the model memorises patterns that do not
transfer to held-out text.

This is the strongest single piece of evidence that the resource difference is about data
*quality and diversity*, not quantity — Model L had more tokens, fit them better, and still
generalised worse.

### Corroborating signals

- **Vocabulary utilisation** (Phase 1): Sindhi's tokenizer showed higher held-out
  utilisation, consistent with a narrower distribution of text.
- **Attention sink**: Model L parks 1.4× more mass on the default position.
- **Repetition under sharpened decoding**: Model L repeats 0.513 at T=0.5 against Model H's
  0.352, consistent with heavier reliance on memorised phrasing.

---

## 8. Deliverables

| Item | Location |
|---|---|
| Transformer implementation | `common/model/` |
| Model configuration files | `hindi/configs/model_config.json`, `sindhi/configs/model_config.json` |
| Training scripts | `common/train/`, `scripts/train_local.sh`, `kaggle/train_kaggle.py` |
| Training logs | `<language>/train/run/train.log` |
| Loss curves (raw) | `<language>/train/run/training_log.jsonl` |
| Loss curves (figure) | `figures/training_curves.png` |
| PPL / BPB tables | `phase2_intrinsic.md`, `intrinsic_results.json` |
| BLEU / chrF / ROUGE-L | `phase2_generation.md`, `generation_results.json` |
| Generated samples + diversity | `phase2_generation.md` |
| Attention heatmaps, entropy, distance | `phase2_attention.md`, `figures/attention_*.png` |
| Resource-level comparison | §7 above |

### Pretrained checkpoints (Google Drive)

- **Model H (Hindi) pretrained checkpoint —** `https://drive.google.com/drive/folders/10fy4e-8qppn0Xd9gEAXP00pZov9xOGzR?usp=sharing`
- **Model L (Sindhi) pretrained checkpoint —** `https://drive.google.com/drive/folders/10fy4e-8qppn0Xd9gEAXP00pZov9xOGzR?usp=sharing`


Local paths, if reproducing: `hindi/train/run/best.pt`, `sindhi/train/run/best.pt`
(~280 MB each). Each checkpoint contains weights, optimizer state, scheduler position, step
count and the model config.

---

## 9. Reproduction

```bash
# 1. Tokenize the corpora (Phase 1 output -> uint16 .bin)
python -m common.data.tokenize_corpus --language hindi
python -m common.data.tokenize_corpus --language sindhi

# 2. Generate the model configs
python -m common.model.make_configs

# 3. Pretrain (resumable; re-run the same command after any interruption)
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

**Tests: 89 across model, data, training and evaluation.**
Run any with `python -m common.<package>.tests.test_<name>`.

---

## 10. Known limitations

1. **Contaminated test splits not cleaned.** Mojibake was measured (7.3% / 16.6%) and
   documented, but not removed, and results were not recomputed on cleaned splits. The
   validation-split cross-check bounds its impact at ~2 percentage points.
2. **No cross-split fuzzy deduplication.** Exact deduplication was global, but MinHash ran
   within each split rather than across them, so some train↔held-out near-duplicates may
   remain and perplexity may be slightly optimistic.
3. **No hyperparameter search.** One run per model with conservative standard values; there
   was no compute budget for a sweep.
4. **Sindhi trained on 81% of its corpus**, to match Hindi's token budget.
5. **The optional no-positional-embeddings ablation was not run.**
