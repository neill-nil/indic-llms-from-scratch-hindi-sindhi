# Sindhi Tokenizer Statistics

This document summarizes the vocabulary and evaluation metrics for the Sindhi tokenizers (BPE and Unigram) trained on the 394M token preprocessed corpus. Both tokenizers were trained from scratch without using any pretrained models or vocabularies.

## 1. Overview
* **Algorithm Chosen for LM Training:** Unigram
* **Target Vocabulary Size:** 32,000

## 2. Unigram Model Statistics (Optimal)
The Unigram algorithm outperformed standard BPE by achieving superior compression (lower fertility) on the held-out validation set.

* **Vocabulary Size:** 32,000 slots
* **Special Tokens:** 261 (including `<s>`, `</s>`, `<pad>`, `<unk>`, and `<|endoftext|>`)
* **Byte Fallback Tokens:** 0
* **Average Characters per Token:** 4.50 characters
* **Target Script Ratio:** 92.59% of the vocabulary successfully mapped to pure Arabic script (Sindhi Perso-Arabic).

### Evaluation Metrics (on held-out set)
* **Fertility:** 1.2454 tokens/word (average)
* **Unknown Token Rate:** 0.0% (Zero `<unk>` tokens generated)

### Tokenization Examples (Unigram)
**Original:** `اٺالو پاندان اپنا هم باز آئي محبت سي`
**Tokens (11):** `['▁اٺ', 'الو', '▁پاند', 'ان', '▁اپ', 'ना', '▁هم', '▁باز', '▁آئي', '▁محبت', '▁سي']`

**Original:** `ماجد ڀرڳڙي ۽ سنڌي ڪمپيوٽنگ`
**Tokens (5):** `['▁ماجد', '▁ڀرڳڙي', '▁۽', '▁سنڌي', '▁ڪمپيوٽنگ']`

## 3. BPE Model Statistics (Baseline)
* **Fertility:** 1.2473 tokens/word (inferior to Unigram)
* **Target Script Ratio:** 93.20%
* **Unknown Token Rate:** 0.0%

### Tokenization Examples (BPE)
**Original:** `ماجد ڀرڳڙي ۽ سنڌي ڪمپيوٽنگ`
**Tokens (5):** `['▁ماجد', '▁ڀرڳڙي', '▁۽', '▁سنڌي', '▁ڪمپيوٽنگ']`
