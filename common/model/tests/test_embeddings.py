"""
Tests for the input representation (Step 1).

Run:
    python -m common.model.tests.test_embeddings
"""

import torch

from common.model.config import GPTConfig
from common.model.embeddings import InputEmbedding


def _cfg(**kw) -> GPTConfig:
    """Small config for fast tests; overrides merge over the defaults."""
    base = dict(vocab_size=100, d_model=32, n_heads=4, d_ff=64, max_seq_len=16, dropout=0.0)
    base.update(kw)
    return GPTConfig(**base)


def test_output_shape():
    """(B, T) ids must produce (B, T, d_model)."""
    cfg = _cfg()
    emb = InputEmbedding(cfg)
    out = emb(torch.randint(0, cfg.vocab_size, (4, 10)))
    assert out.shape == (4, 10, cfg.d_model), out.shape


def test_position_is_actually_injected():
    """
    The whole point of this module: two sequences with the same multiset of tokens in a
    different order must not produce row-wise identical embeddings.

    Without positional embeddings, embedding([a, b]) and embedding([b, a]) would be exact
    row permutations of each other. Adding position breaks that.
    """
    cfg = _cfg()
    emb = InputEmbedding(cfg).eval()
    a = torch.tensor([[5, 9]])
    b = torch.tensor([[9, 5]])
    out_a, out_b = emb(a), emb(b)

    # Row 0 of `a` is token 5 at position 0; row 1 of `b` is token 5 at position 1.
    # If position were absent these would be equal.
    assert not torch.allclose(out_a[0, 0], out_b[0, 1]), \
        "token 5 embeds identically at positions 0 and 1 — position is not being injected"


def test_same_token_different_position_differs():
    """The same token id at two positions must get different vectors."""
    cfg = _cfg()
    emb = InputEmbedding(cfg).eval()
    out = emb(torch.tensor([[7, 7, 7]]))
    assert not torch.allclose(out[0, 0], out[0, 1])
    assert not torch.allclose(out[0, 1], out[0, 2])


def test_position_vector_is_batch_independent():
    """Position t must mean the same thing in every sequence of the batch."""
    cfg = _cfg()
    emb = InputEmbedding(cfg).eval()
    ids = torch.tensor([[3, 4], [3, 4]])
    out = emb(ids)
    assert torch.allclose(out[0], out[1]), "identical sequences embedded differently"


def test_max_seq_len_is_enforced():
    """T > max_seq_len must raise a clear error, not a device-side assert."""
    cfg = _cfg(max_seq_len=8)
    emb = InputEmbedding(cfg)
    emb(torch.zeros(1, 8, dtype=torch.long))  # exactly at the limit is fine
    try:
        emb(torch.zeros(1, 9, dtype=torch.long))
    except ValueError as e:
        assert "exceeds max_seq_len" in str(e), str(e)
    else:
        raise AssertionError("expected ValueError for T > max_seq_len")


def test_rejects_wrong_rank():
    """A (T,) tensor is a common caller mistake and should be caught explicitly."""
    emb = InputEmbedding(_cfg())
    try:
        emb(torch.zeros(10, dtype=torch.long))
    except ValueError as e:
        assert "expected (B, T)" in str(e)
    else:
        raise AssertionError("expected ValueError for 1-D input")


def test_dropout_active_in_train_inactive_in_eval():
    """Dropout must be stochastic in train() and a no-op in eval()."""
    cfg = _cfg(dropout=0.5)
    emb = InputEmbedding(cfg)
    ids = torch.randint(0, cfg.vocab_size, (2, 8))

    emb.train()
    assert not torch.allclose(emb(ids), emb(ids)), "dropout inactive in train mode"

    emb.eval()
    assert torch.allclose(emb(ids), emb(ids)), "dropout still active in eval mode"


def test_init_scale():
    """Embeddings should start at std ~0.02, not the nn.Embedding default of 1.0."""
    cfg = _cfg(vocab_size=4000, d_model=256)
    emb = InputEmbedding(cfg)
    for name, w in [("token", emb.token_embedding.weight),
                    ("position", emb.position_embedding.weight)]:
        std = w.std().item()
        assert 0.015 < std < 0.025, f"{name} embedding std={std:.4f}, expected ~0.02"


def test_gradients_reach_both_tables():
    """Both embedding tables must receive gradient from a downstream loss."""
    cfg = _cfg()
    emb = InputEmbedding(cfg)
    emb(torch.randint(0, cfg.vocab_size, (2, 6))).sum().backward()
    assert emb.token_embedding.weight.grad is not None
    assert emb.position_embedding.weight.grad is not None
    assert emb.position_embedding.weight.grad.abs().sum() > 0


def test_config_rejects_indivisible_heads():
    """d_model must divide by n_heads; catching this at config time beats a shape error."""
    try:
        GPTConfig(d_model=100, n_heads=8)
    except ValueError as e:
        assert "divisible" in str(e)
    else:
        raise AssertionError("expected ValueError for indivisible d_model/n_heads")


def test_analytic_param_count_matches_module():
    """
    The config's analytic count must match the real parameter tree for this stage.

    Guards against the config drifting from the architecture it claims to describe.
    """
    cfg = _cfg(vocab_size=1000, d_model=64, max_seq_len=32)
    emb = InputEmbedding(cfg)
    actual = sum(p.numel() for p in emb.parameters())
    expected = cfg.vocab_size * cfg.d_model + cfg.max_seq_len * cfg.d_model
    assert actual == expected, f"module has {actual:,}, formula says {expected:,}"


def test_target_budget():
    """The shipped config must land near the ~25M target."""
    cfg = GPTConfig(vocab_size=16_000, language="hindi")
    total = cfg.n_params()
    assert 24_000_000 < total < 25_000_000, f"{total:,} outside the ~25M target"
    assert cfg.d_head == 64


def test_no_position_table_when_ablated():
    """The ablation removes the table outright, not by zeroing it."""
    emb = InputEmbedding(_cfg(use_positional_embeddings=False))
    assert emb.position_embedding is None
    assert not any("position" in n for n, _ in emb.named_parameters())


def test_ablation_carries_no_order_information():
    """
    Without positions, the embedding of a token must not depend on where it sits.

    A sequence of one repeated token must embed to identical rows — and, as a control that
    keeps this test from passing vacuously, must NOT do so when positions are present.
    """
    ids = torch.full((1, 8), 7)
    ablated = InputEmbedding(_cfg(use_positional_embeddings=False)).eval()(ids)[0]
    assert torch.equal(ablated, ablated[0].expand_as(ablated)), "rows differ without positions"
    normal = InputEmbedding(_cfg()).eval()(ids)[0]
    assert not torch.equal(normal, normal[0].expand_as(normal)), "control failed"


def test_ablation_param_count_drops_by_exactly_the_table():
    """Analytic and module counts agree, and differ from the full model by max_seq_len x d."""
    from common.model.gpt import GPT
    full, ablated = _cfg(n_layers=1), _cfg(n_layers=1, use_positional_embeddings=False)
    assert GPT(ablated).num_parameters() == ablated.n_params()
    assert full.n_params() - ablated.n_params() == full.max_seq_len * full.d_model


def test_existing_configs_default_to_positions_on():
    """Configs written before the flag existed must keep their positional embeddings."""
    import json, pathlib
    root = pathlib.Path(__file__).resolve().parents[3]
    for lang in ("hindi", "sindhi"):
        raw = json.loads((root / lang / "configs" / "model_config.json").read_text())
        assert "use_positional_embeddings" not in raw
        assert GPTConfig(**raw).use_positional_embeddings is True


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
