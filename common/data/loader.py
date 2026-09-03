"""
Batch loader over a tokenized corpus.

Reads the flat uint16 arrays written by tokenize_corpus.py and serves (inputs, targets)
pairs for causal language-model training.
"""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch

DTYPE = np.uint16


class TokenDataset:
    """
    A memory-mapped view over one split's token stream.

    Why memory-mapping rather than loading
    --------------------------------------
    The Sindhi training split is ~1.24 GB of ids. Loading it into RAM would work here but
    scales badly and competes with the dataloader and the CUDA host allocations. np.memmap
    leaves the data on disk and lets the OS page in only the windows actually touched.
    Because batches are drawn at random the access pattern is scattered, but each window
    is contiguous and small, so the page cache handles it well after the first epoch.

    Why random windows rather than sequential epochs
    -----------------------------------------------
    Training samples a uniformly random start offset for every sequence rather than
    walking the file in order. Two reasons: consecutive windows would be highly correlated
    within a batch, and a fixed window grid would mean a token near a window boundary is
    only ever predicted from a short prefix. Random offsets give every token an
    approximately uniform distribution of available context across the run.

    A consequence worth stating: "one epoch" is not a clean pass over the data. We
    therefore measure training progress in tokens consumed, not epochs.
    """

    def __init__(self, path: Path, block_size: int):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(
                f"{self.path} not found — run common.data.tokenize_corpus first")

        self.block_size = block_size
        self.data = np.memmap(self.path, dtype=DTYPE, mode="r")

        # Each example needs block_size inputs plus one shifted target, so the last legal
        # start offset is len - (block_size + 1).
        self.n_tokens = int(self.data.shape[0])
        self.max_start = self.n_tokens - (block_size + 1)
        if self.max_start < 0:
            raise ValueError(
                f"{self.path.name} holds {self.n_tokens:,} tokens, too few for a "
                f"block size of {block_size}")

    def __len__(self) -> int:
        """Number of distinct start offsets, i.e. distinct training windows."""
        return self.max_start + 1

    def get_batch(
        self,
        batch_size: int,
        device: str = "cpu",
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Draw a random batch.

        Returns:
            x: (batch_size, block_size) inputs
            y: (batch_size, block_size) targets, where y[:, t] is the token following
               x[:, t]. The shift happens here rather than inside the model so that a
               window of block_size+1 tokens yields block_size full training pairs; a
               model-side shift would discard the final position of every window.
        """
        starts = torch.randint(
            0, self.max_start + 1, (batch_size,), generator=generator
        ).tolist()

        # Read each window ONCE as block_size+1 tokens, then take the two overlapping
        # views. Slicing the file twice would double the memmap reads, which dominates
        # when the backing store is slow (a network filesystem, or a cold page cache).
        windows = np.stack([self.data[s: s + self.block_size + 1] for s in starts])
        xs = windows[:, :-1]
        ys = windows[:, 1:]

        # .astype(np.int64) is required: embedding lookups need a long tensor, and torch
        # has no uint16 dtype to convert from directly.
        x = torch.from_numpy(xs.astype(np.int64))
        y = torch.from_numpy(ys.astype(np.int64))

        if device.startswith("cuda"):
            # pin_memory + non_blocking lets the copy overlap with compute.
            x = x.pin_memory().to(device, non_blocking=True)
            y = y.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)
        return x, y

    def iter_sequential(self, batch_size: int, device: str = "cpu"):
        """
        Walk the split in order with non-overlapping windows.

        Evaluation uses this rather than random sampling so that every token is scored
        exactly once and the reported loss is a deterministic function of the split, not
        an estimate that shifts between runs.
        """
        stride = self.block_size
        starts = range(0, self.max_start + 1, stride)
        batch: list[int] = []
        for s in starts:
            batch.append(s)
            if len(batch) == batch_size:
                yield self._gather(batch, device)
                batch = []
        if batch:
            yield self._gather(batch, device)

    def _gather(self, starts, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
        windows = np.stack([self.data[s: s + self.block_size + 1] for s in starts])
        xs = windows[:, :-1]
        ys = windows[:, 1:]
        x = torch.from_numpy(xs.astype(np.int64)).to(device)
        y = torch.from_numpy(ys.astype(np.int64)).to(device)
        return x, y


def load_split(language: str, split: str, block_size: int,
               project_root: Optional[Path] = None) -> TokenDataset:
    """Convenience constructor: load <language>/data/tokenized/<split>.bin."""
    root = project_root or Path(__file__).resolve().parent.parent.parent
    return TokenDataset(root / language / "data" / "tokenized" / f"{split}.bin",
                        block_size=block_size)
