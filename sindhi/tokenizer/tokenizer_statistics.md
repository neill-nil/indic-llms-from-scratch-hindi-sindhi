# Sindhi Tokenizer Statistics

<!-- GENERATED FILE - do not edit by hand.
     Regenerate with:
       python -m common.tokenizer.evaluate_tokenizer --language sindhi --vocab_size 16000
       python -m common.tokenizer.write_stats_report  --language sindhi --vocab_size 16000
-->

Vocabulary and evaluation metrics for the Sindhi tokenizers (Model L (lower-resource)).
Both were trained from scratch with SentencePiece on the preprocessed training corpus;
no pretrained tokenizer or vocabulary was used. All held-out metrics below are measured
on `sindhi/data/preprocessed/val.txt`, which no tokenizer was trained on.

## 1. Selected configuration

* **Algorithm selected for LM training:** Unigram
* **Vocabulary size:** 16,000
* **Character coverage:** 0.9995
* **Byte fallback:** enabled (256 byte tokens), which is why the
  unknown-token rate is exactly zero — any character absent from the vocabulary is
  representable as a sequence of raw byte tokens rather than collapsing to `<unk>`.
* **Special tokens:** 5 (`<pad>`, `<unk>`, `<s>`, `</s>`, `<|endoftext|>`)

### Why 16,000 slots

Vocabulary size trades compression against parameter budget, and at a ~25M parameter
target that budget is the binding constraint. With a model dimension of 512 and a tied
input/output embedding, a 16K vocabulary costs 8.19M parameters;
a 32K vocabulary would cost 16.38M — roughly two-thirds of the entire budget
spent on a lookup table, leaving too little for the transformer stack itself.

The measured cost of that choice is real but modest: a 32K Unigram model compresses this
corpus 4.4% better (held-out fertility 1.2412 vs
1.2987). Paying that to free 8.2M parameters for
depth and feed-forward width is the right trade at this scale.

## 2. Tokenizer training data

Trained on a shuffled reservoir sample of **500,000 lines** from the
cleaned training split (`input_sentence_size`, `shuffle_input_sentence=True`), which bounds
trainer memory on a multi-GB corpus:

| Train split | Sampled for tokenizer | Fraction |
|---|---|---|
| 11,041,567 lines / 618.4M tokens | 500,000 lines / ~28.0M tokens | 4.53% |

**Why this is sufficient.** ~28M tokens against a 16,000-slot
vocabulary is roughly 1,750 observations per entry on average,
well above what is needed to estimate merge frequencies or unigram probabilities. The
held-out numbers below confirm it directly: vocabulary utilisation is
98.2% and the unknown-token rate is exactly zero on text the
tokenizer never saw. A vocabulary fitted to too small a sample would be full of tokens that
never recur.

Sampling applies only to tokenizer training; Phase 2 pretraining uses the entire split.

## 3. Held-out comparison: BPE vs Unigram

Measured on a strided sample across the whole validation file
(21,738 lines, 810,799 words, 1,053,016 tokens).

| Metric | BPE | Unigram | Better | Margin |
|---|---|---|---|---|
| Fertility (tokens / word) | 1.3040 | 1.2987 | Unigram | 0.41% |
| Tokens per character | 0.2772 | 0.2761 | Unigram | 0.40% |
| Bytes per token | 6.383 | 6.409 | Unigram | 0.41% |
| Token entropy (bits/token) | 10.9107 | 10.7703 | Unigram | 1.29% |
| Unknown-token rate | 0.00000000 | 0.00000000 | tie | 0.00% |
| Vocabulary utilisation (higher better) | 97.09% | 98.23% | Unigram | 1.14% pts |
| Target-script share of vocab | 94.49% | 93.51% | BPE | 0.98% pts |

**Verdict.** Unigram is the better choice, but by a narrower margin than raw fertility
alone suggests. The decisive metric is **token entropy**: Unigram produces a token stream
with lower empirical entropy, which directly lower-bounds the perplexity a downstream
language model can achieve on the same text. It also leaves fewer dead embedding rows
(98.2% of its vocabulary appears in held-out text, against
97.1% for BPE) — at a tight parameter budget an unused
vocabulary slot is a wasted embedding vector.

BPE retains a slightly higher target-script share of the vocabulary, but this is not
evidence of quality: the difference is that Unigram allocates more slots to Latin
substrings, and real Sindhi web text contains English loanwords, product names
and numerals that must be encoded either way.

## 4. Vocabulary composition (Unigram)

* **Vocabulary size:** 16,000
* **Perso-Arabic tokens:** 93.51% of non-special vocabulary
* **Script distribution:** arabic 14,717, latin 999, symbol 20, other 2
* **Average token length:** 4.21 characters
* **Median token length:** 4 characters

## 5. Token-frequency statistics (held-out)

* **Distinct token types observed:** 15,717 of 16,000 (98.23%)
* **Token entropy:** 10.7703 bits/token
* **Types covering 50% of all tokens:** 231
* **Types covering 90% of all tokens:** 5,257
* **Types covering 99% of all tokens:** 12,764
* **Unknown-token rate:** 0.00000000

The distribution is steeply Zipfian: just 231 token types account
for half of all token occurrences, while the last 1% of probability mass is spread across
roughly 3,236 rare types.

## 6. Where the two algorithms actually differ

These examples are mined automatically from held-out text by
`mine_contrastive_examples()` rather than chosen by hand, and the comparison is run in
both directions so neither algorithm is shown only its wins. Across frequent word types,
Unigram segments 24 more efficiently and BPE segments 27 more efficiently.

**Words that UNIGRAM segments into fewer tokens (24 distinct frequent word types):**

| Word | Corpus freq. | Unigram | BPE |
|---|---|---|---|
| `جديديت` | 136 | `▁جديديت` | `▁جدي | ديت` |
| `دلاور` | 125 | `▁دلاور` | `▁دلا | ور` |
| `ماٿري` | 119 | `▁ماٿري` | `▁ماٿ | ري` |
| `سُونهن` | 110 | `▁سُونهن` | `▁سُون | هن` |
| `ماتلي` | 91 | `▁ماتلي` | `▁مات | لي` |

**Words that BPE segments into fewer tokens (27 distinct frequent word types):**

| Word | Corpus freq. | Unigram | BPE |
|---|---|---|---|
| `جيون` | 308 | `▁جي | ون` | `▁جيون` |
| `آهين` | 162 | `▁آهي | ن` | `▁آهين` |
| `آهين،` | 148 | `▁آهي | ن | ،` | `▁آهين | ،` |
| `محبتن` | 130 | `▁محبت | ن` | `▁محبتن` |
| `ساريو` | 125 | `▁ساري | و` | `▁ساريو` |


## 7. Sample tokenizations (Unigram)

**Original:** `اٺالو پاندان اپنا هم باز آئي محبت سي`

**Tokens (11):** `▁اٺ | الو | ▁پاند | ان | ▁اپ | نا | ▁هم | ▁باز | ▁آئي | ▁محبت | ▁سي`

**Original:** `ماجد ڀرڳڙي ۽ سنڌي ڪمپيوٽنگ`

**Tokens (8):** `▁ماجد | ▁ڀرڳڙي | ▁۽ | ▁سنڌي | ▁ڪم | پي | و | ٽنگ`

**Original:** `مين چاهتا ٿا اسي روڪ رکون`

**Tokens (7):** `▁مين | ▁چاهت | ا | ▁ٿا | ▁اسي | ▁روڪ | ▁رکون`
