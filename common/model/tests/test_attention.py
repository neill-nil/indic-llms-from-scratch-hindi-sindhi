"""
Tests for multi-head causal self-attention (Step 2).

The centrepiece is test_future_tokens_cannot_leak, which is the empirical verification the
brief asks for: change token t+1 and show the output at position t is unchanged.

Run:
    python -m common.model.tests.test_attention
"""

import math

import torch

from common.model.config import GPTConfig
from common.model.attention import CausalSelfAttention
from common.model.embeddings import InputEmbedding


def _cfg(**kw) -> GPTConfig:
    base = dict(vocab_size=100, d_model=32, n_heads=4, d_ff=64, max_seq_len=16, dropout=0.0)
    base.update(kw)
    return GPTConfig(**base)


# ─────────────────────────────────────────────────────────────────────────────
# Causality — the property the whole mask exists to guarantee
# ─────────────────────────────────────────────────────────────────────────────

def test_future_tokens_cannot_leak():
    """
    Perturb the input at positions > t and assert outputs at positions <= t are bitwise
    unchanged. This is the empirical check the brief requires.

    Uses the real embedding + attention stack so it exercises the actual path, and runs
    in eval() so dropout cannot mask a genuine difference with noise.
    """
    torch.manual_seed(0)
    cfg = _cfg(max_seq_len=12)
    emb, attn = InputEmbedding(cfg).eval(), CausalSelfAttention(cfg).eval()

    T = 10
    ids = torch.randint(0, cfg.vocab_size, (1, T))
    with torch.no_grad():
        base = attn(emb(ids))

    # Change every token strictly after t, for every t.
    for t in range(T - 1):
        altered = ids.clone()
        altered[0, t + 1:] = (altered[0, t + 1:] + 37) % cfg.vocab_size
        assert not torch.equal(altered, ids), "perturbation was a no-op"

        with torch.no_grad():
            new = attn(emb(altered))

        # Positions 0..t must be identical — not merely close.
        assert torch.equal(base[0, :t + 1], new[0, :t + 1]), (
            f"future leaked: changing tokens after position {t} altered output at "
            f"positions 0..{t} (max delta "
            f"{(base[0, :t+1] - new[0, :t+1]).abs().max().item():.2e})"
        )
        # And the change must actually reach position t+1, or the test proves nothing.
        assert not torch.equal(base[0, t + 1], new[0, t + 1]), (
            f"perturbing token {t+1} did not change the output there; the test is vacuous"
        )


def test_attention_weights_are_lower_triangular():
    """No probability mass may sit above the diagonal."""
    cfg = _cfg()
    attn = CausalSelfAttention(cfg).eval()
    _, probs = attn(torch.randn(2, 9, cfg.d_model), return_attention=True)
    upper = torch.triu(torch.ones(9, 9, dtype=torch.bool), diagonal=1)
    assert probs[..., upper].abs().max().item() == 0.0, "attention leaks to future keys"


def test_first_position_attends_only_to_itself():
    """Query 0 has exactly one legal key, so its row must be a one-hot at index 0."""
    cfg = _cfg()
    attn = CausalSelfAttention(cfg).eval()
    _, probs = attn(torch.randn(1, 7, cfg.d_model), return_attention=True)
    row0 = probs[0, :, 0, :]                      # (h, T)
    assert torch.allclose(row0[:, 0], torch.ones_like(row0[:, 0])), row0[:, 0]
    assert row0[:, 1:].abs().max().item() == 0.0


def test_rows_sum_to_one():
    """Softmax over keys: every query's weights form a distribution."""
    cfg = _cfg()
    attn = CausalSelfAttention(cfg).eval()
    _, probs = attn(torch.randn(3, 11, cfg.d_model), return_attention=True)
    sums = probs.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-6), sums.min().item()


def test_no_nan_from_masking():
    """-inf masking must not produce NaN; the diagonal keeps every row populated."""
    cfg = _cfg()
    attn = CausalSelfAttention(cfg).eval()
    out, probs = attn(torch.randn(2, 16, cfg.d_model) * 50, return_attention=True)
    assert torch.isfinite(out).all(), "non-finite values in attention output"
    assert torch.isfinite(probs).all(), "non-finite attention probabilities"


# ─────────────────────────────────────────────────────────────────────────────
# Shapes and the head split
# ─────────────────────────────────────────────────────────────────────────────

def test_output_shape_preserved():
    """Attention is shape-preserving on the residual stream."""
    cfg = _cfg()
    attn = CausalSelfAttention(cfg)
    x = torch.randn(4, 13, cfg.d_model)
    assert attn(x).shape == x.shape


def test_attention_matrix_shape():
    """Returned weights must be (B, n_heads, T, T)."""
    cfg = _cfg(d_model=48, n_heads=6)
    attn = CausalSelfAttention(cfg).eval()
    _, probs = attn(torch.randn(2, 10, cfg.d_model), return_attention=True)
    assert probs.shape == (2, 6, 10, 10), probs.shape


def test_heads_partition_the_channel_axis():
    """
    The (B,T,d) -> (B,h,T,d_head) view must slice channels contiguously, so head i holds
    dims [i*d_head:(i+1)*d_head]. This pins down the reshape convention.
    """
    B, T, h, d_head = 2, 5, 4, 8
    d = h * d_head
    x = torch.arange(B * T * d, dtype=torch.float32).view(B, T, d)
    split = x.view(B, T, h, d_head).transpose(1, 2)      # (B,h,T,d_head)
    for i in range(h):
        assert torch.equal(split[:, i], x[..., i * d_head:(i + 1) * d_head])
    # And the merge is an exact inverse.
    merged = split.transpose(1, 2).contiguous().view(B, T, d)
    assert torch.equal(merged, x), "head merge is not the inverse of the split"


def test_scale_is_inverse_sqrt_d_head():
    """The 1/sqrt(d_k) factor must use the per-head width, not d_model."""
    cfg = _cfg(d_model=64, n_heads=8)
    attn = CausalSelfAttention(cfg)
    assert cfg.d_head == 8
    assert math.isclose(attn.scale, 1 / math.sqrt(8), rel_tol=1e-9), attn.scale


def test_scaling_keeps_scores_in_softmax_range():
    """
    Empirical justification for the scaling: without it, score variance grows with d_head
    and softmax saturates. With it, variance stays ~1.
    """
    torch.manual_seed(0)
    B, h, T, d_head = 1, 1, 64, 64
    q, k = torch.randn(B, h, T, d_head), torch.randn(B, h, T, d_head)
    unscaled = (q @ k.transpose(-2, -1))
    scaled = unscaled / math.sqrt(d_head)
    assert unscaled.std().item() > 4.0, unscaled.std().item()
    assert 0.5 < scaled.std().item() < 2.0, scaled.std().item()


# ─────────────────────────────────────────────────────────────────────────────
# Plumbing
# ─────────────────────────────────────────────────────────────────────────────

def test_sequence_longer_than_mask_raises():
    """T beyond max_seq_len must raise clearly, not index out of bounds."""
    cfg = _cfg(max_seq_len=8)
    attn = CausalSelfAttention(cfg)
    attn(torch.randn(1, 8, cfg.d_model))
    try:
        attn(torch.randn(1, 9, cfg.d_model))
    except ValueError as e:
        assert "exceeds" in str(e), str(e)
    else:
        raise AssertionError("expected ValueError for T > max_seq_len")


def test_mask_is_buffer_not_parameter():
    """The mask must not be trainable, and must follow .to(device)."""
    cfg = _cfg()
    attn = CausalSelfAttention(cfg)
    names = {n for n, _ in attn.named_parameters()}
    assert not any("causal_mask" in n for n in names), "mask registered as a parameter"
    assert "causal_mask" in dict(attn.named_buffers())


def test_dropout_train_vs_eval():
    """Both dropouts active in train(), neither in eval()."""
    cfg = _cfg(dropout=0.5)
    attn = CausalSelfAttention(cfg)
    x = torch.randn(2, 8, cfg.d_model)
    attn.train()
    assert not torch.allclose(attn(x), attn(x)), "dropout inactive during training"
    attn.eval()
    assert torch.allclose(attn(x), attn(x)), "dropout still active in eval"


def test_gradients_reach_all_projections():
    """Q, K, V and W_O must all receive gradient."""
    cfg = _cfg()
    attn = CausalSelfAttention(cfg)
    attn(torch.randn(2, 6, cfg.d_model)).sum().backward()
    for name in ["q_proj", "k_proj", "v_proj", "out_proj"]:
        g = getattr(attn, name).weight.grad
        assert g is not None and g.abs().sum() > 0, f"{name} received no gradient"


def test_parameter_count():
    """4 projections of (d x d + d) each."""
    cfg = _cfg(d_model=64)
    attn = CausalSelfAttention(cfg)
    actual = sum(p.numel() for p in attn.parameters())
    assert actual == 4 * (64 * 64 + 64), actual


def test_batch_independence():
    """Sequences in a batch must not influence each other."""
    torch.manual_seed(0)
    cfg = _cfg()
    attn = CausalSelfAttention(cfg).eval()
    a, b = torch.randn(1, 7, cfg.d_model), torch.randn(1, 7, cfg.d_model)
    with torch.no_grad():
        together = attn(torch.cat([a, b], dim=0))
        alone_a = attn(a)
    assert torch.allclose(together[0], alone_a[0], atol=1e-6)


if __name__ == "__main__":
    torch.manual_seed(0)
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except Exception as e:
            print(f"  FAIL  {t.__name__}: {e}")
            failed += 1
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)
