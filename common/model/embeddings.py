"""
Input representation: token embeddings + positional embeddings.

Stage 1 of the forward pass. Turns a batch of token ids into the dense vectors the
transformer blocks operate on, and injects position information that attention would
otherwise have no access to.
"""

import torch
import torch.nn as nn

from common.model.config import GPTConfig


class InputEmbedding(nn.Module):
    """
    Map token ids to position-aware dense vectors.

    Shapes
    ------
        input   (B, T)      int64 token ids
        output  (B, T, d)   float embeddings

    Why positional embeddings are needed
    ------------------------------------
    Self-attention computes a weighted sum over positions, and that sum is
    *permutation invariant*: shuffling the input tokens permutes the output rows
    identically but changes nothing about what each row contains. Attention alone
    therefore cannot tell "the dog bit the man" from "the man bit the dog". Position
    has to be injected explicitly, and here it is added to the token embedding before
    the first block so every subsequent layer can use it.

    Which scheme, and why
    ---------------------
    We use *learned absolute* positional embeddings: an nn.Embedding of shape
    (max_seq_len, d_model), one trainable vector per slot. The alternatives were
    sinusoidal encodings and a relative scheme such as RoPE.

    Learned absolute is chosen because it is the simplest thing that works at this
    scale, and because at 512 positions there is ample data to fit the table: the
    corpora are ~500M tokens, so every position index is seen on the order of a million
    times. Sinusoidal encodings mainly earn their keep by extrapolating past the trained
    context, which we do not need; RoPE is stronger for long contexts but adds machinery
    to the attention computation that would be harder to justify line by line.

    How this constrains sequence length
    -----------------------------------
    The position table has exactly `max_seq_len` rows, so the model physically cannot
    represent a position beyond it — index `max_seq_len` is out of bounds for the
    embedding and would raise. This is a hard ceiling, not a soft one: unlike sinusoidal
    encodings, which are defined for any index and merely degrade past the training
    range, a learned table has nothing to look up. `forward` therefore checks T and
    raises a clear error rather than letting the CUDA-side index failure surface as an
    opaque device-side assert.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.max_seq_len = config.max_seq_len

        # One row per vocabulary entry. Under weight tying this same matrix is reused as
        # the output projection, so it is trained by gradients from both ends.
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)

        # One row per position index — unless the no-positional-embeddings ablation asks
        # for none. Then there is no table at all rather than a zeroed one: a zero table would
        # still be a trainable parameter that gradients could revive, and would inflate the
        # parameter count the ablation is compared on.
        self.position_embedding = (nn.Embedding(config.max_seq_len, config.d_model)
                                   if config.use_positional_embeddings else None)

        self.dropout = nn.Dropout(config.dropout)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        """
        GPT-2 style initialisation: normal(0, 0.02).

        The default nn.Embedding init is normal(0, 1), which at d_model=512 produces
        activations with a norm around sqrt(512) ~= 22 entering the first LayerNorm.
        That is survivable but wastes early training pulling the scale back down, so we
        start near the scale the residual stream actually wants.
        """
        if isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """
        Embed a batch of token id sequences.

        Args:
            idx: (B, T) int64 tensor of token ids.

        Returns:
            (B, T, d_model) float tensor: token embedding + position embedding, dropped out.
        """
        if idx.dim() != 2:
            raise ValueError(f"expected (B, T) token ids, got shape {tuple(idx.shape)}")

        _, T = idx.shape
        if T > self.max_seq_len:
            raise ValueError(
                f"sequence length {T} exceeds max_seq_len {self.max_seq_len}; the learned "
                f"position table has no row for index {self.max_seq_len} or beyond."
            )

        tok = self.token_embedding(idx)          # (B, T, d)
        if self.position_embedding is None:
            # Ablation: token identity only. The causal mask is now the sole source of order.
            return self.dropout(tok)

        # (T,) position indices on the same device as the input, so this works under
        # .to('cuda') without the caller having to pass anything extra.
        positions = torch.arange(T, device=idx.device)
        pos = self.position_embedding(positions)  # (T, d)

        # pos broadcasts over the batch: (B,T,d) + (T,d) -> (B,T,d). Every sequence in the
        # batch gets the same position vector at a given index, which is what we want —
        # position t means the same thing regardless of which sequence it is in.
        return self.dropout(tok + pos)
