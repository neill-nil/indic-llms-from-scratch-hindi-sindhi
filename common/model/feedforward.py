"""
Position-wise feed-forward network.

The second sublayer of every transformer block. Attention moves information *between*
positions; this network is where each position does its own processing on what it just
gathered. It is "position-wise" because the same two matrices are applied independently to
every position — it mixes channels, never positions, which is what keeps the block causal.
"""

import torch
import torch.nn as nn

from common.model.config import GPTConfig


class FeedForward(nn.Module):
    """
    Two linear layers with a GELU in between: d_model -> d_ff -> d_model.

    Shapes
    ------
        input   (B, T, d_model)
        hidden  (B, T, d_ff)      expanded
        output  (B, T, d_model)   projected back

    Why expand and contract
    -----------------------
    The inner dimension is larger than the model dimension (1536 vs 512, a 3x expansion).
    The width is where this sublayer's capacity lives: it projects into a higher-dimensional
    space, applies a non-linearity there, then projects back. Without the expansion the
    layer would be a bottleneck; without the non-linearity in between the two matrices
    would collapse into a single linear map (W2 @ W1 is just another matrix) and the whole
    sublayer would add nothing that attention's output projection cannot already do.

    Why GELU
    --------
    GELU rather than ReLU because it is smooth: ReLU's derivative is exactly zero for all
    negative inputs, so a unit driven negative receives no gradient and can stay dead.
    GELU has a small negative lobe and is differentiable everywhere, so those units keep
    receiving signal. We use the exact erf-based formulation rather than the tanh
    approximation GPT-2 shipped; the approximation exists for speed on hardware where erf
    was slow, which is not a concern here.

    Why 3x rather than the conventional 4x
    --------------------------------------
    The parameter budget is ~25M. A 4x expansion at d_model=512 would push the model to
    27.4M, roughly 9% over target. The feed-forward ratio is the knob we chose to give up,
    because depth (6 layers) and head dimension (64) matter more at this scale than
    feed-forward width. 1536 is still 3x the residual width, so the sublayer is comfortably
    an expansion rather than a bottleneck.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        if config.d_ff <= config.d_model:
            raise ValueError(
                f"d_ff ({config.d_ff}) must exceed d_model ({config.d_model}); the "
                f"feed-forward inner dimension is meant to be an expansion."
            )
        self.fc_in = nn.Linear(config.d_model, config.d_ff)
        self.act = nn.GELU()
        self.fc_out = nn.Linear(config.d_ff, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, d_model)

        Returns:
            (B, T, d_model)
        """
        x = self.fc_in(x)      # (B, T, d_ff)
        x = self.act(x)
        x = self.fc_out(x)     # (B, T, d_model)
        return self.dropout(x)
