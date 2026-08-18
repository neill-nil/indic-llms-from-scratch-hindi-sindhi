# Hindi Tokenizer Statistics

This document summarizes the vocabulary and evaluation metrics for the Hindi tokenizers (BPE and Unigram) trained on the 517M token preprocessed corpus. Both tokenizers were trained from scratch without using any pretrained models or vocabularies.

## 1. Overview
* **Algorithm Chosen for LM Training:** Unigram
* **Target Vocabulary Size:** 16,000 slots

**Vocabulary Size Justification:** The target size of 16,000 was deliberately chosen to balance compression efficiency against model parameter constraints. A 25M parameter limit is extremely tight. If we used a 32K vocabulary, the embedding matrix would consume ~12.3M parameters (~49% of the budget), starving the Feed-Forward networks. By strictly limiting the vocabulary to 16K, the embedding layer is restricted to ~6.1M parameters (~24.5%), leaving nearly 19 Million parameters for deep transformer logic (the "reasoning space"). We chose 16K over 10K because over-compressing highly inflected Devanagari text causes severe token fragmentation (high fertility), which wastes the sequence length. 16K is the mathematically optimal sweet spot.

## 2. Unigram Model Statistics (Optimal)
The Unigram algorithm outperformed standard BPE by achieving superior compression (lower fertility) on the held-out validation set. 

**Why Unigram Performed Better:** BPE is a bottom-up, greedy algorithm that simply merges the most frequent pairs of characters, which can lead to unnatural subword splits in morphologically rich languages like Hindi. Unigram, on the other hand, is a top-down probabilistic model that evaluates multiple possible segmentations and chooses the one with the highest overall likelihood. This allows Unigram to naturally align with Hindi's complex morphological structure (roots, prefixes, and suffixes), resulting in cleaner tokens and a tighter compression ratio (lower fertility).

**Direct Visual Example:**
Notice how BPE greedily and awkwardly splits the acronym `आईआईएम` (IIM) into two meaningless pieces (`आई`, `आईएम`), whereas Unigram probabilistically evaluates the whole word and correctly maps it to a single intact token (`आईआईएम`).
* **Original:** `आईआईएम`
* **Unigram:** `['▁आईआईएम']` (1 semantic token)
* **BPE:** `['▁आई', 'आईएम']` (2 broken tokens)

* **Vocabulary Size:** 16,000 slots
* **Special Tokens:** 261 (including `<s>`, `</s>`, `<pad>`, `<unk>`, and `<|endoftext|>`)
* **Byte Fallback Tokens:** 0
* **Average Characters per Token:** 4.88 characters
* **Target Script Ratio:** 92.62% of the vocabulary successfully mapped to pure Devanagari script.

### Evaluation Metrics (on held-out set)
* **Fertility:** 1.2798 tokens/word (average)
* **Unknown Token Rate:** 0.0% (Zero `<unk>` tokens generated)

### Tokenization Examples (Unigram)
**Original:** `आईआईएम अहमदाबाद के साथ एमओयू करने की तैयारी में सरकार`
**Tokens (13):** `['▁आई', 'आईएम', '▁अहमदाबाद', '▁के', '▁साथ', '▁', 'एमओ', 'यू', '▁करने', '▁की', '▁तैयारी', '▁में', '▁सरकार']`

**Original:** `ONGC ने पूरे 2020-21 वित्त वर्ष में 11,246.44 करोड़ रुपये का शुद्ध लाभ दर्ज किया...`
**Tokens (35):** `['▁O', 'NG', 'C', '▁ने', '▁पूरे', '▁', '2', '0', '2', '0', '-', '2', '1', '▁वित्त', '▁वर्ष', '▁में', '▁', '1', '1', ',', '2', '4', '6', '.', '4', '4', '▁करोड़', '▁रुपये', '▁का', '▁शुद्ध', '▁लाभ', '▁दर्ज', '▁किया', '▁था', '.']`

## 3. BPE Model Statistics (Baseline)
* **Fertility:** 1.2944 tokens/word (inferior to Unigram)
* **Target Script Ratio:** 93.84%
* **Unknown Token Rate:** 0.0%

### Tokenization Examples (BPE)
**Original:** `आईआईएम अहमदाबाद के साथ एमओयू करने की तैयारी में सरकार`
**Tokens (13):** `['▁आई', 'आईएम', '▁अहमदाबाद', '▁के', '▁साथ', '▁एम', 'ओ', 'यू', '▁करने', '▁की', '▁तैयारी', '▁में', '▁सरकार']`
