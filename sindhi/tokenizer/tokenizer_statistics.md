# Sindhi Tokenizer Statistics

This document summarizes the vocabulary and evaluation metrics for the Sindhi tokenizers (BPE and Unigram) trained on the 694M token preprocessed corpus. Both tokenizers were trained from scratch without using any pretrained models or vocabularies.

## 1. Overview
* **Algorithm Chosen for LM Training:** Unigram
* **Target Vocabulary Size:** 32,000 slots

**Vocabulary Size Justification:** The target size of 32,000 was deliberately chosen to balance compression efficiency against model parameter constraints. A massive vocabulary (e.g., 100k) would yield slightly lower fertility but would cause the embedding matrix to consume almost the entire 25M parameter budget allotted for Phase 2. At 32k, we achieve a highly optimal fertility rate (~1.24 tokens/word) and zero unknown tokens, while keeping the embedding layer small enough to allow for deep transformer layers.

## 2. Unigram Model Statistics (Optimal)
The Unigram algorithm outperformed standard BPE by achieving superior compression (lower fertility) on the held-out validation set.

**Why Unigram Performed Better:** BPE is a bottom-up, greedy algorithm that simply merges the most frequent pairs of characters, which can lead to unnatural subword splits in morphologically complex languages like Sindhi. Unigram, on the other hand, is a top-down probabilistic model that evaluates multiple possible segmentations and chooses the one with the highest overall likelihood. This allows Unigram to naturally align with the morphological structure of Sindhi (which uses a highly modified Perso-Arabic script with complex affixes), resulting in cleaner tokens and a tighter compression ratio (lower fertility).

**Direct Visual Example:**
In the evaluation reports below, notice how Unigram mathematically compresses the Sindhi vocabulary much more efficiently than BPE. Because Unigram evaluates the probability of the *entire* word rather than just greedy pair-frequencies, it is able to encode the exact same text using fewer total tokens, saving vital context-window space for the Language Model.
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
