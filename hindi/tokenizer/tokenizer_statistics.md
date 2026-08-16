# Hindi Tokenizer Statistics

This document summarizes the vocabulary and evaluation metrics for the Hindi tokenizers (BPE and Unigram) trained on the 523M token preprocessed corpus. Both tokenizers were trained from scratch without using any pretrained models or vocabularies.

## 1. Overview
* **Algorithm Chosen for LM Training:** Unigram
* **Target Vocabulary Size:** 32,000

## 2. Unigram Model Statistics (Optimal)
The Unigram algorithm outperformed standard BPE by achieving superior compression (lower fertility) on the held-out validation set.

* **Vocabulary Size:** 32,000 slots
* **Special Tokens:** 261 (including `<s>`, `</s>`, `<pad>`, `<unk>`, and `<|endoftext|>`)
* **Byte Fallback Tokens:** 0
* **Average Characters per Token:** 5.21 characters
* **Target Script Ratio:** 91.34% of the vocabulary successfully mapped to pure Devanagari script.

### Evaluation Metrics (on held-out set)
* **Fertility:** 1.2214 tokens/word (average)
* **Unknown Token Rate:** 0.0% (Zero `<unk>` tokens generated)

### Tokenization Examples (Unigram)
**Original:** `आईआईएम अहमदाबाद के साथ एमओयू करने की तैयारी में सरकार`
**Tokens (10):** `['▁आईआईएम', '▁अहमदाबाद', '▁के', '▁साथ', '▁एमओयू', '▁करने', '▁की', '▁तैयारी', '▁में', '▁सरकार']`

**Original:** `ONGC ने पूरे 2020-21 वित्त वर्ष में 11,246.44 करोड़ रुपये का शुद्ध लाभ दर्ज किया था.`
**Tokens (35):** `['▁O', 'NG', 'C', '▁ने', '▁पूरे', '▁', '2', '0', '2', '0', '-', '2', '1', '▁वित्त', '▁वर्ष', '▁में', '▁', '1', '1', ',', '2', '4', '6', '.', '4', '4', '▁करोड़', '▁रुपये', '▁का', '▁शुद्ध', '▁लाभ', '▁दर्ज', '▁किया', '▁था', '.']`

## 3. BPE Model Statistics (Baseline)
* **Fertility:** 1.2276 tokens/word (inferior to Unigram)
* **Target Script Ratio:** 92.22%
* **Unknown Token Rate:** 0.0%

### Tokenization Examples (BPE)
**Original:** `आईआईएम अहमदाबाद के साथ एमओयू करने की तैयारी में सरकार`
**Tokens (11):** `['▁आई', 'आईएम', '▁अहमदाबाद', '▁के', '▁साथ', '▁एमओयू', '▁करने', '▁की', '▁तैयारी', '▁में', '▁सरकार']`
