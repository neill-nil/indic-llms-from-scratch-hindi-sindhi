# Phase 2: Intrinsic Language-Modelling Evaluation

<!-- GENERATED FILE - do not edit by hand.
     Regenerate with:
       python -m common.eval.intrinsic
       python -m common.eval.write_intrinsic_report
-->

Cross-entropy, perplexity and bits-per-byte for Model H (Hindi) and Model L (Sindhi) on
their own held-out splits. Every token in each split is scored exactly once, using
non-overlapping windows, so these numbers are deterministic rather than sampled estimates.

## Results

| Model | Split | Context | CE (nats) | Perplexity | BPB | Tokens | Bytes |
|---|---|---|---|---|---|---|---|
| Hindi | val | 512 (native) | 3.4929 | 32.88 | **0.5056** | 14,825,984 | 147,755,937 |
| Hindi | test | 512 (native) | 3.5722 | 35.59 | **0.5375** | 11,826,176 | 113,388,968 |
| Sindhi | val | 1024 (native) | 3.9541 | 52.15 | **0.8821** | 18,111,488 | 117,131,189 |
| Sindhi | val | 512 (matched) | 3.9960 | 54.38 | **0.8915** | 18,112,000 | 117,119,126 |
| Sindhi | test | 1024 (native) | 2.9940 | 19.97 | **0.9503** | 14,955,520 | 67,981,284 |
| Sindhi | test | 512 (matched) | 3.0194 | 20.48 | **0.9584** | 14,956,032 | 67,978,756 |

Model L is evaluated twice: at its native 1024-token context, and again at 512 to match
Model H exactly. The matched rows are what the cross-model comparison rests on, because
a longer context lowers loss for free and would otherwise flatter Model L.

## Headline comparison

At **matched context (512 tokens) on the test splits**:

| | Hindi (H) | Sindhi (L) | gap |
|---|---|---|---|
| Cross-entropy | 3.5722 | 3.0194 | — |
| Perplexity | 35.59 | 20.48 | — |
| **Bits per byte** | **0.5375** | **0.9584** | **+78.3%** |

Model L needs **78.3% more bits per byte** of text than Model H. Both models have
identical transformer stacks, identical vocabulary sizes, identical token budgets and
identical effective batches, so this gap reflects the languages and their corpora rather
than the training setup.

## Why bits-per-byte, not perplexity, is the headline metric

Perplexity is per *token*, and the two models have separate tokenizers learned on separate
corpora. A token means a different amount of text in each, so the numbers are not directly
comparable. Bits-per-byte normalises by the underlying UTF-8 bytes and is therefore
tokenizer-independent, which is why the brief asks for it.

This corpus provides an unusually sharp demonstration. On Sindhi, perplexity says the test
split is **0.4x easier** than validation (19.97 vs
52.15), which is not credible for two halves of the same collection.
Bits-per-byte says the opposite and correct thing: test is slightly *harder*
(0.9503 vs 0.8821). The explanation is in
the next section — perplexity was misled by tokenization, BPB was not.

## Corpus contamination, and its effect on these numbers

Investigating the perplexity anomaly above revealed that both test splits contain
encoding-corrupted text: legacy non-Unicode Sindhi and Hindi fonts decoded as Unicode,
producing strings of valid script characters in nonsense order, typically with digits
wedged inside words. A representative Sindhi example:

```
ـام ١اال ١اك ٧ي ٻٽي٨٦ ٧ؾار ةيؽك حيٝ رهزا٤٦ س٩٧ي ١ام ٤ڪؽي ٿ٩ۿ
```

Measured rates, over 40,000 documents per split:

| | val | test |
|---|---|---|
| Hindi | 0.6% | **7.3%** |
| Sindhi | 1.6% | **16.6%** |

This corrupted text is short, highly repetitive and therefore easy to predict, which
deflates per-token perplexity wherever it is dense. It arrives as ~80-character fragments,
which is why the Sindhi test split holds 208,300 documents against validation's 53,252
despite having fewer tokens.

**Why the Phase 1 filters did not catch it.** `script_filter` required at least 90% of
alphabetic characters to fall in the expected Unicode block. Mojibake passes that check
comfortably: it *is* Devanagari or Arabic-block characters, just in meaningless sequence.
Detecting it needs a different signal — here, the density of digits appearing inside
words, which legitimate prose in either language almost never produces.

**Effect on the reported numbers.** The two test splits are contaminated at different
rates (7.3% vs 16.6%), so some part of the headline BPB gap is contamination asymmetry
rather than language difficulty. The validation splits are far cleaner (0.6% and 1.6%) and
are the better basis for a like-for-like comparison; at matched 512-token context the
validation BPB gap is
76.3%,
against 78.3% on test. Both figures support the same conclusion, and the
validation figure is the more conservative of the two.

## Method

* Windows are non-overlapping and stride by the context length, so every token is scored
  exactly once and the result is a deterministic function of the split.
* Cross-entropy is the mean over all scored positions, in nats.
* Perplexity is `exp(CE)`.
* Bits-per-byte is `CE * n_tokens / (ln(2) * n_bytes)`, converting from nats-per-token to
  bits-per-byte using the true UTF-8 byte count of the underlying text.
* Byte counts come from the `.txt` splits the `.bin` files were tokenized from, excluding
  the document separator lines that carry no text.

Raw measurements: `report/intrinsic_results.json`.
