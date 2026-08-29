"""
Multi-head causal self-attention, implemented from first principles.

Stage 2 of the forward pass. Every tensor operation here is explicit: no
nn.MultiheadAttention, no F.scaled_dot_product_attention, no fused kernels. Besides the
project constraint, writing the softmax out by hand is what lets us return the attention
probability matrix for the Phase 2 attention analysis (heatmaps, per-head entropy, mean
attention distance) — a fused kernel never materialises it.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.model.config import GPTConfig


class CausalSelfAttention(nn.Module):
    """
    Masked multi-head self-attention over a sequence.

    Shapes, end to end
    ------------------
        x              (B, T, d)            input residual stream
        q, k, v        (B, T, d)            three independent linear projections
        after reshape  (B, h, T, d_head)    h heads, each in a d/h subspace
        scores         (B, h, T, T)         query-key affinity per head
        probs          (B, h, T, T)         row-softmaxed, causally masked
        context        (B, h, T, d_head)    probs @ v
        after merge    (B, T, d)            heads concatenated
        output         (B, T, d)            after W_O

    The reshape and transpose
    -------------------------
    A projection gives (B, T, d) where d is one flat vector per position. Splitting into
    heads is a *view*: `.view(B, T, h, d_head)` reinterprets the last axis as h contiguous
    slices of width d_head without moving data. Head i owns dimensions
    [i*d_head : (i+1)*d_head] — the heads partition the channel axis, they do not
    duplicate it, which is why h heads cost the same parameters as one.

    The following `.transpose(1, 2)` gives (B, h, T, d_head), putting h next to the batch
    axis so the T x d_head matrices batch cleanly through matmul: torch treats all leading
    axes as batch, so one `q @ k.transpose(-2,-1)` computes every head of every sequence
    at once.

    Merging back reverses it: `.transpose(1, 2)` returns (B, T, h, d_head), then
    `.contiguous().view(B, T, d)` lays the heads back out side by side. The
    `.contiguous()` is required — transpose only permutes strides, and `view` refuses a
    non-contiguous tensor. Skipping it raises; using `.reshape()` instead would silently
    copy. Concatenating heads this way is exactly the Concat(head_1..head_h) of the paper.

    Why scores are divided by sqrt(d_head)
    --------------------------------------
    If the components of q and k are roughly independent with unit variance, their dot
    product over d_head dimensions is a sum of d_head such products, so it has variance
    d_head and typical magnitude sqrt(d_head). At d_head=64 raw scores would run to ~8 in
    absolute value, and softmax over values that large is effectively an argmax: one
    probability saturates near 1 and the rest near 0. In that regime the softmax Jacobian
    is nearly zero, so almost no gradient flows back to q and k and the head stops
    learning. Dividing by sqrt(d_head) rescales the scores back to unit variance, keeping
    softmax in its responsive range. It is a variance correction, not a temperature knob.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.d_model = config.d_model
        self.d_head = config.d_head
        # Precomputed because it is used every forward pass.
        self.scale = 1.0 / math.sqrt(self.d_head)

        # Q = X W_Q, K = X W_K, V = X W_V. Kept as three separate projections rather than
        # one fused (d -> 3d) matrix: identical parameter count, and it mirrors the
        # formulation directly.
        self.q_proj = nn.Linear(config.d_model, config.d_model)
        self.k_proj = nn.Linear(config.d_model, config.d_model)
        self.v_proj = nn.Linear(config.d_model, config.d_model)

        # W_O, applied after the heads are concatenated.
        self.out_proj = nn.Linear(config.d_model, config.d_model)

        # Two distinct dropouts, as the brief requires: one on the attention
        # probabilities (drops individual query-key edges) and one on the sublayer output
        # (drops channels of what gets added back to the residual stream).
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Lower-triangular mask, 1 where attention is allowed. Registered as a buffer so
        # it moves with .to(device) and is saved/loaded with the module, but is not a
        # parameter and receives no gradient. Built once at max_seq_len and sliced per
        # call, so no allocation happens in the forward pass.
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config.max_seq_len, config.max_seq_len, dtype=torch.bool)),
            persistent=False,
        )

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        """
        Args:
            x: (B, T, d_model) residual stream.
            return_attention: also return the (B, n_heads, T, T) probability matrix.
                Used by the attention analysis; off during training to save memory.

        Returns:
            (B, T, d_model), or a tuple with the attention probabilities appended.
        """
        B, T, d = x.shape
        if d != self.d_model:
            raise ValueError(f"expected last dim {self.d_model}, got {d}")
        if T > self.causal_mask.size(0):
            raise ValueError(
                f"sequence length {T} exceeds the {self.causal_mask.size(0)}-position "
                f"causal mask built for max_seq_len"
            )

        # ── Project ──────────────────────────────────────────────────────────────
        q = self.q_proj(x)   # (B, T, d)
        k = self.k_proj(x)   # (B, T, d)
        v = self.v_proj(x)   # (B, T, d)

        # ── Split into heads: (B, T, d) -> (B, h, T, d_head) ─────────────────────
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # ── Scaled dot-product scores ────────────────────────────────────────────
        # (B,h,T,d_head) @ (B,h,d_head,T) -> (B,h,T,T). Entry [b,h,i,j] is how much
        # query i attends to key j.
        scores = (q @ k.transpose(-2, -1)) * self.scale

        # ── Causal mask ──────────────────────────────────────────────────────────
        # Additive mask in spirit: 0 on allowed positions, -inf on future ones. We apply
        # it with masked_fill rather than adding a matrix of zeros and -infs, which is
        # numerically identical and avoids materialising the addend.
        #
        # -inf is safe here specifically because the mask is lower-triangular with a
        # populated diagonal: every query attends to at least itself, so no softmax row
        # is fully masked. A fully-masked row would softmax to NaN.
        mask = self.causal_mask[:T, :T]            # (T, T) bool
        scores = scores.masked_fill(~mask, float("-inf"))

        # ── Normalise and mix ────────────────────────────────────────────────────
        # Softmax over the key axis, so each query's weights over keys sum to 1.
        probs = F.softmax(scores, dim=-1)          # (B, h, T, T)
        probs = self.attn_dropout(probs)

        context = probs @ v                        # (B, h, T, d_head)

        # ── Merge heads and project ──────────────────────────────────────────────
        # (B,h,T,d_head) -> (B,T,h,d_head) -> (B,T,d). .contiguous() is mandatory:
        # transpose only permutes strides and view requires contiguous memory.
        context = context.transpose(1, 2).contiguous().view(B, T, self.d_model)

        out = self.resid_dropout(self.out_proj(context))

        if return_attention:
            # Return the post-softmax, pre-dropout-semantics weights the analysis wants.
            # In eval mode dropout is a no-op, so `probs` is exactly the attention
            # distribution; the analysis always runs under eval.
            return out, probs
        return out
