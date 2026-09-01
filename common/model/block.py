"""
The transformer block: attention + feed-forward, each wrapped in a residual connection
with pre-layer-normalisation.

Stage 3 of the forward pass. N of these are stacked to form the model body.
"""

import math

import torch
import torch.nn as nn

from common.model.config import GPTConfig
from common.model.attention import CausalSelfAttention
from common.model.feedforward import FeedForward


class TransformerBlock(nn.Module):
    """
    One decoder block.

        x = x + Attention(LayerNorm(x))
        x = x + FeedForward(LayerNorm(x))

    Shapes
    ------
        input and output are both (B, T, d_model). The block is shape-preserving, which is
        what lets N of them stack.

    Pre-norm, and why
    -----------------
    We place the LayerNorm *inside* the residual branch, before each sublayer. The
    alternative, post-norm, is what the original Transformer paper used:

        pre-norm   x = x + Sublayer(LN(x))
        post-norm  x = LN(x + Sublayer(x))

    The difference is what happens to the residual stream. Under pre-norm the stream is a
    clean additive path from the embedding all the way to the final norm: every block only
    ever *adds* to it. Backpropagating along that path passes through addition, whose local
    gradient is exactly 1, so gradient reaches the early layers undiminished no matter how
    deep the stack is.

    Under post-norm every block's output passes *through* a LayerNorm before continuing, so
    the gradient from the loss to layer 1 is a product of N LayerNorm Jacobians. Those
    Jacobians rescale by roughly 1/sigma of the activations, and the product of N such
    factors drifts exponentially with depth. In practice post-norm stacks need a careful
    learning-rate warmup to avoid diverging in the first few hundred steps, and the deeper
    they get the more fragile that becomes. Pre-norm is what GPT-2 onwards uses, and it is
    the reason a 6-layer stack here trains without any warmup gymnastics.

    Why a final LayerNorm is needed after the stack
    -----------------------------------------------
    The flip side of the clean additive highway is that nothing ever normalises it. Each
    block adds a contribution, so the stream's magnitude grows with depth. The model body
    therefore ends with one more LayerNorm (applied in the full model, not here) to bring
    the stream back to a sane scale before the output projection. Under post-norm that
    final norm would be redundant, since the last block already normalised.

    Why causality survives this block
    ---------------------------------
    Attention is the only operation here that moves information between positions, and it
    is masked. LayerNorm normalises across the channel axis within a single position; the
    feed-forward network applies the same two matrices independently at each position; the
    residual add is elementwise. None of them look sideways, so a block cannot leak the
    future even though only one of its four components is explicitly masked.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_attn = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ln_ffn = nn.LayerNorm(config.d_model)
        self.ffn = FeedForward(config)

        self.apply(self._init_weights)
        self._scale_residual_projections(config.n_layers)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        """GPT-2 style: linear weights normal(0, 0.02), biases zero."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def _scale_residual_projections(self, n_layers: int) -> None:
        """
        Shrink the two projections that write into the residual stream.

        Every block makes two additive contributions to the stream, so across N blocks the
        stream accumulates 2N of them. Sums of independent contributions grow like the
        square root of their count, so without correction the activations entering the
        final norm scale like sqrt(2N) and deeper models start out progressively hotter.

        Scaling the *writing* projections — attention's output projection and the
        feed-forward's second layer — by 1/sqrt(2N) cancels that growth, so the stream
        starts at a stable magnitude regardless of depth. This is the GPT-2 initialisation
        trick, and it is why deep stacks can skip the warmup that post-norm needs.
        """
        std = 0.02 / math.sqrt(2 * n_layers)
        for projection in (self.attn.out_proj, self.ffn.fc_out):
            nn.init.normal_(projection.weight, mean=0.0, std=std)

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        """
        Args:
            x: (B, T, d_model)
            return_attention: also return this block's (B, n_heads, T, T) attention weights.

        Returns:
            (B, T, d_model), or a tuple with the attention weights appended.
        """
        # Sublayer 1: attention. Note the norm is applied to the branch input only — the
        # value added back to `x` is the raw sublayer output, and `x` itself is untouched.
        if return_attention:
            attn_out, attn_weights = self.attn(self.ln_attn(x), return_attention=True)
        else:
            attn_out, attn_weights = self.attn(self.ln_attn(x)), None
        x = x + attn_out

        # Sublayer 2: position-wise feed-forward.
        x = x + self.ffn(self.ln_ffn(x))

        if return_attention:
            return x, attn_weights
        return x
