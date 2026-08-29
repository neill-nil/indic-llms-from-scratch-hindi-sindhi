# Hindi Tokenizer Statistics

<!-- GENERATED FILE - do not edit by hand.
     Regenerate with:
       python -m common.tokenizer.evaluate_tokenizer --language hindi --vocab_size 16000
       python -m common.tokenizer.write_stats_report  --language hindi --vocab_size 16000
-->

Vocabulary and evaluation metrics for the Hindi tokenizers (Model H (higher-resource)).
Both were trained from scratch with SentencePiece on the preprocessed training corpus;
no pretrained tokenizer or vocabulary was used. All held-out metrics below are measured
on `hindi/data/preprocessed/val.txt`, which no tokenizer was trained on.

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
corpus 4.7% better (held-out fertility 1.2437 vs
1.3056). Paying that to free 8.2M parameters for
depth and feed-forward width is the right trade at this scale.

## 2. Tokenizer training data

Trained on a shuffled reservoir sample of **500,000 lines** from the
cleaned training split (`input_sentence_size`, `shuffle_input_sentence=True`), which bounds
trainer memory on a multi-GB corpus:

| Train split | Sampled for tokenizer | Fraction |
|---|---|---|
| 7,238,248 lines / 504.8M tokens | 500,000 lines / ~34.9M tokens | 6.91% |

**Why this is sufficient.** ~35M tokens against a 16,000-slot
vocabulary is roughly 2,179 observations per entry on average,
well above what is needed to estimate merge frequencies or unigram probabilities. The
held-out numbers below confirm it directly: vocabulary utilisation is
97.4% and the unknown-token rate is exactly zero on text the
tokenizer never saw. A vocabulary fitted to too small a sample would be full of tokens that
never recur.

Sampling applies only to tokenizer training; Phase 2 pretraining uses the entire split.

## 3. Held-out comparison: BPE vs Unigram

Measured on a strided sample across the whole validation file
(10,960 lines, 665,138 words, 868,410 tokens).

| Metric | BPE | Unigram | Better | Margin |
|---|---|---|---|---|
| Fertility (tokens / word) | 1.3203 | 1.3056 | Unigram | 1.11% |
| Tokens per character | 0.2598 | 0.2569 | Unigram | 1.12% |
| Bytes per token | 9.805 | 9.915 | Unigram | 1.11% |
| Token entropy (bits/token) | 10.6594 | 10.5347 | Unigram | 1.17% |
| Unknown-token rate | 0.00000000 | 0.00000000 | tie | 0.00% |
| Vocabulary utilisation (higher better) | 95.52% | 97.39% | Unigram | 1.87% pts |
| Target-script share of vocab | 93.84% | 92.62% | BPE | 1.23% pts |

**Verdict.** Unigram is the better choice, but by a narrower margin than raw fertility
alone suggests. The decisive metric is **token entropy**: Unigram produces a token stream
with lower empirical entropy, which directly lower-bounds the perplexity a downstream
language model can achieve on the same text. It also leaves fewer dead embedding rows
(97.4% of its vocabulary appears in held-out text, against
95.5% for BPE) — at a tight parameter budget an unused
vocabulary slot is a wasted embedding vector.

BPE retains a slightly higher target-script share of the vocabulary, but this is not
evidence of quality: the difference is that Unigram allocates more slots to Latin
substrings, and real Hindi web text contains English loanwords, product names
and numerals that must be encoded either way.

## 4. Vocabulary composition (Unigram)

* **Vocabulary size:** 16,000
* **Devanagari tokens:** 92.62% of non-special vocabulary
* **Script distribution:** devanagari 14,576, latin 1,152, symbol 7, other 2, arabic 1
* **Average token length:** 4.88 characters
* **Median token length:** 5 characters

## 5. Token-frequency statistics (held-out)

* **Distinct token types observed:** 15,583 of 16,000 (97.39%)
* **Token entropy:** 10.5347 bits/token
* **Types covering 50% of all tokens:** 174
* **Types covering 90% of all tokens:** 5,054
* **Types covering 99% of all tokens:** 12,545
* **Unknown-token rate:** 0.00000000

The distribution is steeply Zipfian: just 174 token types account
for half of all token occurrences, while the last 1% of probability mass is spread across
roughly 3,455 rare types.

## 6. Where the two algorithms actually differ

These examples are mined automatically from held-out text by
`mine_contrastive_examples()` rather than chosen by hand, and the comparison is run in
both directions so neither algorithm is shown only its wins. Across frequent word types,
Unigram segments 88 more efficiently and BPE segments 34 more efficiently.

**Words that UNIGRAM segments into fewer tokens (88 distinct frequent word types):**

| Word | Corpus freq. | Unigram | BPE |
|---|---|---|---|
| `ज्वालामुखी` | 96 | `▁ज्वालामुखी` | `▁ज्वाल | ामुखी` |
| `सेमेस्टर` | 86 | `▁सेमेस्टर` | `▁से | मे | स्टर` |
| `द्विआधारी` | 85 | `▁द्विआधारी` | `▁द्वि | आ | धारी` |
| `साईं` | 82 | `▁साईं` | `▁साई | ं` |
| `कोरबा` | 82 | `▁कोरबा` | `▁कोर | बा` |

**Words that BPE segments into fewer tokens (34 distinct frequent word types):**

| Word | Corpus freq. | Unigram | BPE |
|---|---|---|---|
| `है।'` | 198 | `▁है | । | '` | `▁है | ।'` |
| `Also` | 111 | `▁ | Al | so` | `▁Al | so` |
| `कांग्रेसी` | 96 | `▁कांग्रेस | ी` | `▁कांग्रेसी` |
| `केयर` | 91 | `▁के | यर` | `▁केयर` |
| `हैं।'` | 84 | `▁हैं | । | '` | `▁हैं | ।'` |


## 7. Sample tokenizations (Unigram)

**Original:** `आईआईएम अहमदाबाद के साथ एमओयू करने की तैयारी में सरकार`

**Tokens (13):** `▁आई | आईएम | ▁अहमदाबाद | ▁के | ▁साथ | ▁ | एमओ | यू | ▁करने | ▁की | ▁तैयारी | ▁में | ▁सरकार`

**Original:** `8:25 PM बोल्ट ने पहले ओवर में दिए चार रन, जिसमें ओवर की आखिरी शोर्ट पिच गेंद पर इमाम ने मारा शानदार चौका।`

**Tokens (30):** `▁ | 8 | : | 2 | 5 | ▁PM | ▁बोल्ट | ▁ने | ▁पहले | ▁ओवर | ▁में | ▁दिए | ▁चार | ▁रन | , | ▁जिसमें | ▁ओवर | ▁की | ▁आखिरी | ▁शो | र्ट | ▁पिच | ▁गेंद | ▁पर`

**Original:** `ONGC ने पूरे 2020-21 वित्त वर्ष में 11,246.44 करोड़ रुपये का शुद्ध लाभ दर्ज किया था.`

**Tokens (35):** `▁O | NG | C | ▁ने | ▁पूरे | ▁ | 2 | 0 | 2 | 0 | - | 2 | 1 | ▁वित्त | ▁वर्ष | ▁में | ▁ | 1 | 1 | , | 2 | 4 | 6 | .`
