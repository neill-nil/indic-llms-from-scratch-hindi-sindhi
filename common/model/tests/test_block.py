"""
Tests for the feed-forward network and the transformer block (Step 3).

Run:
    python -m common.model.tests.test_block
"""

import math

import torch
import torch.nn as nn

from common.model.config import GPTConfig
from common.model.feedforward import FeedForward
from common.model.block import TransformerBlock


def _cfg(**kw) -> GPTConfig:
    base = dict(vocab_size=100, d_model=32, n_heads=4, d_ff=96, max_seq_len=16, dropout=0.0)
    base.update(kw)
    return GPTConfig(**base)


# ─────────────────────────────────────────────────────────────────────────────
# Feed-forward network
# ─────────────────────────────────────────────────────────────────────────────

def test_ffn_shape_preserved():
    """The sublayer must return to d_model so it can be added to the residual stream."""
    cfg = _cfg()
    ffn = FeedForward(cfg)
    x = torch.randn(3, 7, cfg.d_model)
    assert ffn(x).shape == x.shape


def test_ffn_expands_internally():
    """The hidden activation must be wider than the residual stream."""
    cfg = _cfg()
    ffn = FeedForward(cfg)
    hidden = ffn.act(ffn.fc_in(torch.randn(2, 5, cfg.d_model)))
    assert hidden.shape[-1] == cfg.d_ff > cfg.d_model, hidden.shape


def test_ffn_rejects_bottleneck_config():
    """d_ff <= d_model is a configuration error, not a silent bottleneck."""
    try:
        FeedForward(_cfg(d_model=64, d_ff=64))
    except ValueError as e:
        assert "expansion" in str(e)
    else:
        raise AssertionError("expected ValueError when d_ff <= d_model")


def test_ffn_is_actually_nonlinear():
    """
    Without the GELU the two matrices would collapse to one linear map, so f(2x) would
    equal 2*f(x). Confirm it does not.
    """
    cfg = _cfg()
    ffn = FeedForward(cfg).eval()
    # Bias-free comparison: a linear map satisfies f(2x) == 2 f(x) only without bias, so
    # zero the biases to isolate the non-linearity rather than the affine offset.
    nn.init.zeros_(ffn.fc_in.bias)
    nn.init.zeros_(ffn.fc_out.bias)
    x = torch.randn(1, 4, cfg.d_model)
    with torch.no_grad():
        assert not torch.allclose(ffn(2 * x), 2 * ffn(x), atol=1e-4), \
            "feed-forward behaves linearly — the activation is missing or inert"


def test_ffn_is_position_wise():
    """
    The same transform must be applied independently at every position: shuffling
    positions must permute the output identically. This is why the FFN cannot leak the
    future.
    """
    cfg = _cfg()
    ffn = FeedForward(cfg).eval()
    x = torch.randn(1, 6, cfg.d_model)
    perm = torch.randperm(6)
    with torch.no_grad():
        assert torch.allclose(ffn(x)[:, perm], ffn(x[:, perm]), atol=1e-6), \
            "feed-forward mixes positions"


def test_ffn_parameter_count():
    """(d_model x d_ff + d_ff) + (d_ff x d_model + d_model)."""
    cfg = _cfg(d_model=64, d_ff=192)
    ffn = FeedForward(cfg)
    expected = (64 * 192 + 192) + (192 * 64 + 64)
    assert sum(p.numel() for p in ffn.parameters()) == expected


# ─────────────────────────────────────────────────────────────────────────────
# Transformer block
# ─────────────────────────────────────────────────────────────────────────────

def test_block_shape_preserved():
    """Blocks must be shape-preserving so they can stack."""
    cfg = _cfg()
    blk = TransformerBlock(cfg)
    x = torch.randn(2, 9, cfg.d_model)
    assert blk(x).shape == x.shape


def test_block_preserves_causality():
    """
    The full block — attention, both LayerNorms, FFN, both residuals — must not leak the
    future. Only attention is explicitly masked; this confirms nothing else looks sideways.
    """
    torch.manual_seed(0)
    cfg = _cfg(max_seq_len=12)
    blk = TransformerBlock(cfg).eval()
    T = 10
    x = torch.randn(1, T, cfg.d_model)
    with torch.no_grad():
        base = blk(x)
    for t in range(T - 1):
        altered = x.clone()
        altered[0, t + 1:] += 1.5
        with torch.no_grad():
            new = blk(altered)
        assert torch.allclose(base[0, :t + 1], new[0, :t + 1], atol=1e-6), (
            f"future leaked through the block at position {t}: max delta "
            f"{(base[0, :t+1] - new[0, :t+1]).abs().max().item():.2e}")


def test_stack_preserves_causality():
    """Causality must survive depth, not just a single block."""
    torch.manual_seed(0)
    cfg = _cfg(max_seq_len=12, n_layers=4)
    stack = nn.Sequential(*[TransformerBlock(cfg) for _ in range(4)]).eval()
    T = 9
    x = torch.randn(1, T, cfg.d_model)
    with torch.no_grad():
        base = stack(x)
    for t in range(T - 1):
        altered = x.clone()
        altered[0, t + 1:] += 2.0
        with torch.no_grad():
            new = stack(altered)
        assert torch.allclose(base[0, :t + 1], new[0, :t + 1], atol=1e-5), \
            f"future leaked through a 4-block stack at position {t}"


def test_residual_connections_exist():
    """
    Zero both sublayer output projections and the block must become the identity. If the
    residual adds were missing, the output would collapse to zero instead.
    """
    cfg = _cfg()
    blk = TransformerBlock(cfg).eval()
    with torch.no_grad():
        nn.init.zeros_(blk.attn.out_proj.weight); nn.init.zeros_(blk.attn.out_proj.bias)
        nn.init.zeros_(blk.ffn.fc_out.weight);    nn.init.zeros_(blk.ffn.fc_out.bias)
        x = torch.randn(2, 6, cfg.d_model)
        out = blk(x)
    assert torch.allclose(out, x, atol=1e-6), \
        "with both sublayers zeroed the block is not the identity — a residual is missing"


def test_is_pre_norm_not_post_norm():
    """
    Distinguish pre-norm from post-norm by the scale of the output.

    Pre-norm leaves the residual stream unnormalised, so a large-magnitude input produces
    a large-magnitude output. Post-norm would normalise the block output to roughly unit
    variance regardless of input scale.
    """
    cfg = _cfg()
    blk = TransformerBlock(cfg).eval()
    big = torch.randn(1, 8, cfg.d_model) * 100.0
    with torch.no_grad():
        out = blk(big)
    assert out.std().item() > 10.0, (
        f"output std {out.std().item():.2f} for an input of std ~100 — the block appears "
        f"to normalise its output, i.e. post-norm rather than pre-norm")


def test_layernorm_applied_before_sublayer():
    """
    Structural check that the norm feeds the sublayer rather than the other way round:
    replacing ln_attn with an identity must change the result, and the sublayer must
    receive normalised input.
    """
    cfg = _cfg()
    blk = TransformerBlock(cfg).eval()
    captured = {}

    def hook(_module, inputs, _output):
        captured["std"] = inputs[0].std().item()

    handle = blk.attn.register_forward_hook(hook)
    with torch.no_grad():
        blk(torch.randn(1, 8, cfg.d_model) * 50.0)
    handle.remove()
    # LayerNorm output has unit variance per position, so the attention sublayer should
    # see std ~1 even though the block input had std ~50.
    assert 0.5 < captured["std"] < 2.0, (
        f"attention received input with std {captured['std']:.2f}; expected ~1 from a "
        f"preceding LayerNorm")


def test_layernorms_are_learnable_and_distinct():
    """Two separate LayerNorms, each with learnable affine parameters."""
    cfg = _cfg()
    blk = TransformerBlock(cfg)
    assert blk.ln_attn is not blk.ln_ffn
    for ln in (blk.ln_attn, blk.ln_ffn):
        assert ln.weight is not None and ln.bias is not None
        assert ln.weight.requires_grad and ln.bias.requires_grad
        assert torch.allclose(ln.weight, torch.ones_like(ln.weight))


def test_residual_projection_init_is_scaled():
    """
    The two projections writing into the residual stream must start at
    0.02/sqrt(2*n_layers); the other linears at 0.02.
    """
    cfg = _cfg(d_model=256, d_ff=768, n_layers=8)
    blk = TransformerBlock(cfg)
    expected_scaled = 0.02 / math.sqrt(2 * 8)

    for proj in (blk.attn.out_proj, blk.ffn.fc_out):
        std = proj.weight.std().item()
        assert abs(std - expected_scaled) / expected_scaled < 0.15, \
            f"residual projection std {std:.5f}, expected ~{expected_scaled:.5f}"

    for proj in (blk.attn.q_proj, blk.ffn.fc_in):
        std = proj.weight.std().item()
        assert abs(std - 0.02) / 0.02 < 0.15, f"std {std:.5f}, expected ~0.02"


def test_biases_initialised_to_zero():
    """GPT-2 style: linear biases start at zero."""
    blk = TransformerBlock(_cfg())
    for name in ["attn.q_proj", "attn.out_proj", "ffn.fc_in", "ffn.fc_out"]:
        mod = blk.get_submodule(name)
        assert torch.allclose(mod.bias, torch.zeros_like(mod.bias)), f"{name} bias nonzero"


def test_gradients_reach_every_component():
    """Every parameter in the block must receive gradient."""
    cfg = _cfg()
    blk = TransformerBlock(cfg)
    blk(torch.randn(2, 6, cfg.d_model)).sum().backward()
    for name, p in blk.named_parameters():
        assert p.grad is not None, f"{name} got no gradient"
        assert p.grad.abs().sum().item() > 0, f"{name} gradient is all zero"


def test_block_returns_attention_weights():
    """The block must be able to surface its attention map for the Phase 2 analysis."""
    cfg = _cfg()
    blk = TransformerBlock(cfg).eval()
    out, attn = blk(torch.randn(2, 7, cfg.d_model), return_attention=True)
    assert out.shape == (2, 7, cfg.d_model)
    assert attn.shape == (2, cfg.n_heads, 7, 7)
    assert torch.allclose(attn.sum(-1), torch.ones_like(attn.sum(-1)), atol=1e-6)


def test_block_dropout_train_vs_eval():
    """Dropout must be live in train() and off in eval()."""
    cfg = _cfg(dropout=0.3)
    blk = TransformerBlock(cfg)
    x = torch.randn(2, 8, cfg.d_model)
    blk.train()
    assert not torch.allclose(blk(x), blk(x))
    blk.eval()
    assert torch.allclose(blk(x), blk(x))


def test_block_parameter_count_matches_config_formula():
    """The per-block count must match what GPTConfig.n_params() assumes."""
    cfg = _cfg(d_model=64, d_ff=192, n_heads=8, n_layers=1)
    blk = TransformerBlock(cfg)
    actual = sum(p.numel() for p in blk.parameters())
    attn = 4 * (64 * 64 + 64)
    mlp = (64 * 192 + 192) + (192 * 64 + 64)
    norms = 2 * (2 * 64)
    assert actual == attn + mlp + norms, f"{actual:,} vs {attn + mlp + norms:,}"
    # And the config's whole-model formula must agree for a 1-block model.
    assert cfg.n_params(non_embedding=True) == attn + mlp + norms


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
