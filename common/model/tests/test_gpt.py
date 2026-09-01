"""
Tests for the assembled model (Step 4).

Run:
    python -m common.model.tests.test_gpt
"""

import math
from pathlib import Path

import torch

from common.model.config import GPTConfig
from common.model.gpt import GPT

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def _cfg(**kw) -> GPTConfig:
    base = dict(vocab_size=100, d_model=32, n_heads=4, d_ff=96,
                max_seq_len=16, dropout=0.0, n_layers=2)
    base.update(kw)
    return GPTConfig(**base)


# ─────────────────────────────────────────────────────────────────────────────
# Shapes and loss
# ─────────────────────────────────────────────────────────────────────────────

def test_logits_shape():
    cfg = _cfg()
    m = GPT(cfg)
    logits, loss = m(torch.randint(0, cfg.vocab_size, (3, 9)))
    assert logits.shape == (3, 9, cfg.vocab_size), logits.shape
    assert loss is None, "loss must be None when no targets are given"


def test_loss_matches_manual_cross_entropy():
    """The reported loss must be plain mean cross-entropy over all positions."""
    torch.manual_seed(0)
    cfg = _cfg()
    m = GPT(cfg).eval()
    idx = torch.randint(0, cfg.vocab_size, (2, 7))
    tgt = torch.randint(0, cfg.vocab_size, (2, 7))
    with torch.no_grad():
        logits, loss = m(idx, tgt)
        manual = torch.nn.functional.cross_entropy(
            logits.reshape(-1, cfg.vocab_size), tgt.reshape(-1))
    assert torch.allclose(loss, manual, atol=1e-6)


def test_initial_loss_is_near_uniform_entropy():
    """
    An untrained model should be roughly uniform over the vocabulary, so the loss should
    start near ln(vocab_size). A value far from this means the initialisation is wrong —
    for example, badly scaled embeddings producing confidently wrong predictions.
    """
    torch.manual_seed(0)
    cfg = _cfg(vocab_size=16000, d_model=128, d_ff=384, n_layers=2, max_seq_len=64)
    m = GPT(cfg).eval()
    idx = torch.randint(0, cfg.vocab_size, (4, 64))
    tgt = torch.randint(0, cfg.vocab_size, (4, 64))
    with torch.no_grad():
        _, loss = m(idx, tgt)
    expected = math.log(cfg.vocab_size)
    assert abs(loss.item() - expected) < 0.5, \
        f"initial loss {loss.item():.3f}, expected ~ln(V)={expected:.3f}"


# ─────────────────────────────────────────────────────────────────────────────
# Weight tying
# ─────────────────────────────────────────────────────────────────────────────

def test_head_is_the_same_tensor_as_the_embedding():
    """Tying must share one tensor, not copy values that then drift apart."""
    m = GPT(_cfg(tie_embeddings=True))
    assert m.lm_head.weight is m.embeddings.token_embedding.weight, \
        "lm_head.weight is a different object — the weights are copied, not tied"


def test_tied_gradients_accumulate_from_both_paths():
    """
    A tied matrix is used twice per forward pass, so it should receive gradient from both
    the embedding lookup and the output projection.
    """
    cfg = _cfg()
    m = GPT(cfg)
    idx = torch.randint(0, cfg.vocab_size, (2, 6))
    _, loss = m(idx, torch.randint(0, cfg.vocab_size, (2, 6)))
    loss.backward()
    g = m.embeddings.token_embedding.weight.grad
    assert g is not None and g.abs().sum() > 0
    # Rows never seen as input still get gradient, because every row participates in the
    # output projection at every position.
    unseen = sorted(set(range(cfg.vocab_size)) - set(idx.flatten().tolist()))
    assert g[unseen].abs().sum() > 0, \
        "unused vocabulary rows have no gradient — the head does not share the matrix"


def test_tying_saves_exactly_vocab_times_dmodel():
    tied = GPT(_cfg(tie_embeddings=True)).num_parameters()
    untied = GPT(_cfg(tie_embeddings=False)).num_parameters()
    cfg = _cfg()
    assert untied - tied == cfg.vocab_size * cfg.d_model, (untied, tied)


def test_parameters_deduplicates_tied_weight():
    """Counting must not charge for the shared matrix twice."""
    cfg = _cfg(tie_embeddings=True)
    m = GPT(cfg)
    counted = sum(p.numel() for p in m.parameters())
    naive = sum(p.numel() for _, p in m.named_parameters(remove_duplicate=False))
    assert naive - counted == cfg.vocab_size * cfg.d_model, \
        "parameters() did not de-duplicate the tied matrix"


# ─────────────────────────────────────────────────────────────────────────────
# Parameter count
# ─────────────────────────────────────────────────────────────────────────────

def test_param_count_matches_analytic_formula():
    """The real module tree must agree with GPTConfig.n_params()."""
    for kw in [dict(), dict(tie_embeddings=False), dict(n_layers=4, d_model=64, d_ff=192)]:
        cfg = _cfg(**kw)
        m = GPT(cfg)
        assert m.num_parameters() == cfg.n_params(), \
            f"{kw}: module {m.num_parameters():,} vs formula {cfg.n_params():,}"


def test_non_embedding_count_excludes_both_tables():
    cfg = _cfg()
    m = GPT(cfg)
    diff = m.num_parameters() - m.num_parameters(non_embedding=True)
    assert diff == cfg.vocab_size * cfg.d_model + cfg.max_seq_len * cfg.d_model


# ─────────────────────────────────────────────────────────────────────────────
# Causality, end to end
# ─────────────────────────────────────────────────────────────────────────────

def test_logits_at_t_ignore_future_tokens():
    """
    The spec-mandated check, now at the level of the final logits rather than an
    intermediate activation: changing token t+1 must not move the logits at position t.
    """
    torch.manual_seed(0)
    cfg = _cfg(max_seq_len=12, n_layers=3)
    m = GPT(cfg).eval()
    T = 10
    idx = torch.randint(0, cfg.vocab_size, (1, T))
    with torch.no_grad():
        base, _ = m(idx)
    for t in range(T - 1):
        alt = idx.clone()
        alt[0, t + 1:] = (alt[0, t + 1:] + 41) % cfg.vocab_size
        with torch.no_grad():
            new, _ = m(alt)
        assert torch.allclose(base[0, :t + 1], new[0, :t + 1], atol=1e-5), (
            f"logits at positions 0..{t} moved when tokens after {t} changed; max delta "
            f"{(base[0, :t+1] - new[0, :t+1]).abs().max().item():.2e}")


# ─────────────────────────────────────────────────────────────────────────────
# Generation
# ─────────────────────────────────────────────────────────────────────────────

def test_generate_appends_requested_length():
    cfg = _cfg()
    m = GPT(cfg).eval()
    prompt = torch.randint(0, cfg.vocab_size, (2, 4))
    out = m.generate(prompt, max_new_tokens=6)
    assert out.shape == (2, 10)
    assert torch.equal(out[:, :4], prompt), "prompt was modified"
    assert out.min() >= 0 and out.max() < cfg.vocab_size


def test_generate_greedy_is_deterministic():
    """temperature=0 must take the argmax, so repeated calls agree."""
    torch.manual_seed(0)
    cfg = _cfg()
    m = GPT(cfg).eval()
    prompt = torch.randint(0, cfg.vocab_size, (1, 3))
    a = m.generate(prompt, max_new_tokens=5, temperature=0.0)
    b = m.generate(prompt, max_new_tokens=5, temperature=0.0)
    assert torch.equal(a, b), "greedy decoding is not deterministic"


def test_generate_respects_context_window():
    """A prompt longer than max_seq_len must be cropped, not crash."""
    cfg = _cfg(max_seq_len=8)
    m = GPT(cfg).eval()
    long_prompt = torch.randint(0, cfg.vocab_size, (1, 20))
    out = m.generate(long_prompt, max_new_tokens=3)
    assert out.shape == (1, 23)


def test_generate_top_k_restricts_support():
    """With k=1, sampling must coincide with greedy."""
    torch.manual_seed(0)
    cfg = _cfg()
    m = GPT(cfg).eval()
    prompt = torch.randint(0, cfg.vocab_size, (1, 3))
    greedy = m.generate(prompt, max_new_tokens=5, temperature=0.0)
    top1 = m.generate(prompt, max_new_tokens=5, temperature=1.0, top_k=1)
    assert torch.equal(greedy, top1), "top_k=1 should reproduce greedy decoding"


def test_generate_restores_training_mode():
    """generate() flips to eval internally and must put the model back."""
    m = GPT(_cfg())
    m.train()
    m.generate(torch.randint(0, 100, (1, 3)), max_new_tokens=2)
    assert m.training, "generate() left the model in eval mode"


# ─────────────────────────────────────────────────────────────────────────────
# Plumbing
# ─────────────────────────────────────────────────────────────────────────────

def test_attention_maps_returned_per_layer():
    cfg = _cfg(n_layers=3)
    m = GPT(cfg).eval()
    logits, _, attns = m(torch.randint(0, cfg.vocab_size, (2, 7)), return_attention=True)
    assert len(attns) == 3
    for a in attns:
        assert a.shape == (2, cfg.n_heads, 7, 7)
        assert torch.allclose(a.sum(-1), torch.ones_like(a.sum(-1)), atol=1e-6)


def test_all_parameters_receive_gradient():
    cfg = _cfg()
    m = GPT(cfg)
    _, loss = m(torch.randint(0, cfg.vocab_size, (2, 6)),
                torch.randint(0, cfg.vocab_size, (2, 6)))
    loss.backward()
    for name, p in m.named_parameters():
        assert p.grad is not None, f"{name}: no gradient"
        assert p.grad.abs().sum().item() > 0, f"{name}: zero gradient"


def test_shipped_configs_build_and_match_targets():
    """Both real configs must instantiate at the documented parameter counts."""
    expected = {"hindi": (512, 24_220_672), "sindhi": (1024, 24_482_816)}
    non_embedding = set()
    for lang, (ctx, total) in expected.items():
        path = PROJECT_ROOT / lang / "configs" / "model_config.json"
        m = GPT.from_config_file(path)
        assert m.config.max_seq_len == ctx
        assert m.num_parameters() == total, f"{lang}: {m.num_parameters():,} != {total:,}"
        assert 24_000_000 < m.num_parameters() < 25_000_000
        non_embedding.add(m.num_parameters(non_embedding=True))
    assert len(non_embedding) == 1, \
        f"transformer stacks differ between models: {non_embedding}"


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
