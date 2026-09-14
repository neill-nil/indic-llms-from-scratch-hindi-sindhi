"""
Batch loader over the synthetic reasoning corpus.

Presents the same `get_batch(batch_size, device, generator) -> (x, y)` contract as
`TokenDataset`, so the Phase 2 `Trainer` can consume it without modification. The
difference is entirely in what `y` contains: positions that should not be learned carry
-100, which `GPT.forward` passes to `F.cross_entropy(ignore_index=-100)`.

Why the prompt is masked out of the loss
----------------------------------------
An example is a question followed by an answer. The question is always *given* at
inference time, so learning to reproduce it teaches nothing useful — and it is roughly
nine tenths of the tokens, so an unmasked loss would spend nine tenths of the gradient
signal on text the model will never have to produce. Masking concentrates the entire
budget on the handful of tokens that actually constitute the answer.

Why right padding needs no attention mask
-----------------------------------------
Examples have different lengths, so short ones are padded to a common width. With a
causal mask, a padded position can only ever be attended *by positions after it*, and
every position after it is itself padding — which is excluded from the loss. No real
token can see a pad token, so padding cannot influence any gradient. This is why the
attention implementation needed no padding-mask support: the causal mask already makes
right padding harmless.

Why the answer is tokenized with a leading space
------------------------------------------------
SentencePiece marks word beginnings with `▁`, so `"मुंबई"` and `" मुंबई"` tokenize
differently. At inference the model continues a prompt ending in `उत्तर:` and must emit
the *space-prefixed* form. Encoding the answer without that space during training would
teach it a token sequence it can never usefully produce.
"""

import json
import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import sentencepiece as spm

logger = logging.getLogger("finetune.dataset")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

IGNORE_INDEX = -100
DOC_SEPARATOR = "<|endoftext|>"

# Measured across every split of both languages: the longest example is 85 tokens (Sindhi
# test_hop, five entities). 96 clears that with headroom and no example is ever truncated,
# while being well under the 128 first guessed — padding is pure wasted compute, and at a
# mean length of 38-58 tokens there is already a great deal of it.
MAX_LEN = 96


class ReasoningDataset:
    """
    One split of the reasoning corpus, tokenized once and held in memory.

    The corpus is small — 40,000 examples of roughly 60 tokens — so unlike the
    pretraining stream there is no reason to memory-map it. Tokenizing once at startup
    costs a few seconds and removes SentencePiece from the training loop entirely.
    """

    def __init__(self, path: Path, tokenizer_path: Path, max_len: int = MAX_LEN,
                 limit: Optional[int] = None):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(
                f"{self.path} not found — run common.reasoning.generate first")

        self.sp = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
        self.max_len = max_len
        self.pad_id = self.sp.pad_id()
        self.eos_id = self.sp.piece_to_id(DOC_SEPARATOR)

        self.examples: list[dict] = []
        with open(self.path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if limit is not None and i >= limit:
                    break
                self.examples.append(json.loads(line))

        self.x = np.full((len(self.examples), max_len), self.pad_id, dtype=np.int64)
        self.y = np.full((len(self.examples), max_len), IGNORE_INDEX, dtype=np.int64)
        self.answer_starts: list[int] = []
        self.prompt_lengths: list[int] = []

        n_truncated = 0
        for row, ex in enumerate(self.examples):
            prompt_ids = self.sp.encode(ex["prompt"])
            # Leading space: see the module docstring.
            answer_ids = self.sp.encode(" " + ex["answer"]) + [self.eos_id]
            full = prompt_ids + answer_ids

            if len(full) > max_len + 1:
                n_truncated += 1
                # Truncate from the FRONT of the prompt, never the answer. A clipped
                # answer would silently become an unreachable training target.
                full = full[-(max_len + 1):]
                prompt_ids = full[:-len(answer_ids)]

            xs, ys = full[:-1], full[1:]
            self.x[row, :len(xs)] = xs
            self.y[row, :len(ys)] = ys

            # y[t] predicts full[t+1], so the first position whose target is an answer
            # token is t = len(prompt) - 1. Everything before it is prompt and is masked.
            first = max(0, len(prompt_ids) - 1)
            self.y[row, :first] = IGNORE_INDEX
            self.y[row, len(ys):] = IGNORE_INDEX          # padding
            self.answer_starts.append(first)
            self.prompt_lengths.append(len(prompt_ids))

        supervised = int((self.y != IGNORE_INDEX).sum())
        logger.info("%s: %d examples, %d supervised tokens (%.1f%% of %d positions)%s",
                    self.path.name, len(self.examples), supervised,
                    100 * supervised / self.y.size, self.y.size,
                    f", {n_truncated} truncated" if n_truncated else "")
        if n_truncated:
            logger.warning("%d examples exceeded max_len=%d and were front-truncated",
                           n_truncated, max_len)

    def __len__(self) -> int:
        return len(self.examples)

    def get_batch(self, batch_size: int, device: str = "cpu",
                  generator: Optional[torch.Generator] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Draw a random batch, matching TokenDataset's signature exactly."""
        idx = torch.randint(0, len(self.examples), (batch_size,), generator=generator)
        return self._gather(idx.tolist(), device)

    def iter_sequential(self, batch_size: int, device: str = "cpu"):
        """Walk the split in order, so evaluation scores every example exactly once."""
        for start in range(0, len(self.examples), batch_size):
            yield self._gather(list(range(start, min(start + batch_size,
                                                     len(self.examples)))), device)

    def _gather(self, rows: list[int], device: str):
        x = torch.from_numpy(self.x[rows])
        y = torch.from_numpy(self.y[rows])
        if device.startswith("cuda"):
            x = x.pin_memory().to(device, non_blocking=True)
            y = y.pin_memory().to(device, non_blocking=True)
            return x, y
        return x.to(device), y.to(device)


def load_reasoning_split(language: str, split: str, max_len: int = MAX_LEN,
                         limit: Optional[int] = None,
                         project_root: Optional[Path] = None) -> ReasoningDataset:
    """Load <language>/finetune/data/<split>.jsonl with that language's own tokenizer."""
    root = project_root or PROJECT_ROOT
    return ReasoningDataset(
        root / language / "finetune" / "data" / f"{split}.jsonl",
        root / language / "tokenizer" / f"{language}_unigram_16000.model",
        max_len=max_len, limit=limit)
