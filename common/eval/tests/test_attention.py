"""
Tests for the attention statistics (Step 10).

Each metric is checked against a hand-constructed attention pattern whose correct answer
is known analytically, so a wrong formula cannot pass.

Run:
    python -m common.eval.tests.test_attention
"""

import math

import numpy as np
import torch

from common.eval.attention import attention_statistics, sample_clean_windows


def causal(rows):
    """Build a (1, 1, T, T) attention tensor from a list of row distributions."""
    t = torch.tensor(rows, dtype=torch.float32)
    return t.unsqueeze(0).unsqueeze(0)


def uniform_causal(T):
    """Row t spreads mass evenly over keys 0..t — the maximum-entropy causal pattern."""
    a = torch.zeros(T, T)
    for t in range(T):
        a[t, :t + 1] = 1.0 / (t + 1)
    return a.unsqueeze(0).unsqueeze(0)


def diagonal_causal(T, offset=0):
    """All mass on position t-offset: a pure positional head."""
    a = torch.zeros(T, T)
    for t in range(T):
        a[t, max(0, t - offset)] = 1.0
    return a.unsqueeze(0).unsqueeze(0)


# ─────────────────────────────────────────────────────────────────────────────
# Entropy
# ─────────────────────────────────────────────────────────────────────────────

def test_uniform_attention_has_normalised_entropy_one():
    """Spreading evenly over everything visible is by definition maximal entropy."""
    s = attention_statistics(uniform_causal(64), min_query=32)
    assert abs(s["entropy_normalised"][0] - 1.0) < 1e-4, s["entropy_normalised"][0]


def test_point_attention_has_zero_entropy():
    """All mass on one key means no uncertainty."""
    s = attention_statistics(diagonal_causal(64), min_query=32)
    assert s["entropy_bits"][0] < 1e-4
    assert s["entropy_normalised"][0] < 1e-4


def test_entropy_bits_matches_hand_calculation():
    """
    Query 2 splits evenly over 2 of its 3 visible keys -> 1 bit.
    Using min_query=2 with T=3 isolates that single row.
    """
    a = causal([[1.0, 0.0, 0.0],
                [0.5, 0.5, 0.0],
                [0.5, 0.5, 0.0]])
    s = attention_statistics(a, min_query=2)
    assert abs(s["entropy_bits"][0] - 1.0) < 1e-5, s["entropy_bits"][0]
    # Normalised by log2(3) available keys.
    assert abs(s["entropy_normalised"][0] - 1.0 / math.log2(3)) < 1e-5


def test_entropy_ignores_masked_zeros():
    """Zero-probability future keys must not contribute NaN or spurious entropy."""
    s = attention_statistics(uniform_causal(48), min_query=16)
    assert np.isfinite(s["entropy_bits"]).all()
    assert np.isfinite(s["entropy_normalised"]).all()


# ─────────────────────────────────────────────────────────────────────────────
# Mean distance
# ─────────────────────────────────────────────────────────────────────────────

def test_self_attention_has_zero_distance():
    s = attention_statistics(diagonal_causal(64, offset=0), min_query=32)
    assert abs(s["mean_distance"][0]) < 1e-5


def test_previous_token_head_has_distance_one():
    """A head attending only to t-1 must measure exactly 1 token of distance."""
    s = attention_statistics(diagonal_causal(64, offset=1), min_query=32)
    assert abs(s["mean_distance"][0] - 1.0) < 1e-5


def test_fixed_offset_head_measures_that_offset():
    for off in (3, 7, 15):
        s = attention_statistics(diagonal_causal(96, offset=off), min_query=32)
        assert abs(s["mean_distance"][0] - off) < 1e-5, (off, s["mean_distance"][0])


def test_uniform_attention_distance_is_half_the_context():
    """
    Uniform over keys 0..t gives mean distance t/2. Averaged over queries 32..63 of a
    64-length window that is the mean of t/2, i.e. (31.5+ ... )/2.
    """
    T, mq = 64, 32
    s = attention_statistics(uniform_causal(T), min_query=mq)
    expected = np.mean([t / 2 for t in range(mq, T)])
    assert abs(s["mean_distance"][0] - expected) < 1e-4


# ─────────────────────────────────────────────────────────────────────────────
# Positional focus
# ─────────────────────────────────────────────────────────────────────────────

def test_self_and_prev_token_mass():
    s = attention_statistics(diagonal_causal(64, offset=0), min_query=32)
    assert abs(s["self_attention"][0] - 1.0) < 1e-5
    assert abs(s["prev_token"][0]) < 1e-5

    s = attention_statistics(diagonal_causal(64, offset=1), min_query=32)
    assert abs(s["self_attention"][0]) < 1e-5
    assert abs(s["prev_token"][0] - 1.0) < 1e-5


# ─────────────────────────────────────────────────────────────────────────────
# Shape handling and guards
# ─────────────────────────────────────────────────────────────────────────────

def test_per_head_outputs_are_independent():
    """Two heads with different patterns must produce different statistics."""
    a = torch.cat([diagonal_causal(64, 0), uniform_causal(64)], dim=1)   # (1,2,T,T)
    s = attention_statistics(a, min_query=32)
    assert len(s["entropy_bits"]) == 2
    assert s["entropy_bits"][0] < 1e-4                  # point head
    assert s["entropy_bits"][1] > 3.0                   # uniform head
    assert s["mean_distance"][0] < s["mean_distance"][1]


def test_averages_over_batch():
    a = uniform_causal(64).repeat(5, 1, 1, 1)
    one = attention_statistics(uniform_causal(64), min_query=32)
    many = attention_statistics(a, min_query=32)
    assert abs(one["entropy_bits"][0] - many["entropy_bits"][0]) < 1e-5


def test_rejects_wrong_rank_and_non_square():
    for bad in (torch.zeros(4, 4), torch.zeros(1, 2, 8, 9)):
        try:
            attention_statistics(bad, min_query=2)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for shape {tuple(bad.shape)}")


def test_rejects_min_query_beyond_sequence():
    try:
        attention_statistics(uniform_causal(16), min_query=32)
    except ValueError as e:
        assert "min_query" in str(e)
    else:
        raise AssertionError("expected ValueError when min_query >= T")


# ─────────────────────────────────────────────────────────────────────────────
# Window sampling
# ─────────────────────────────────────────────────────────────────────────────

def test_windows_exclude_separators():
    sep = 5
    toks = np.arange(20_000, dtype=np.uint16) % 100
    toks[::197] = sep
    w = sample_clean_windows(toks, sep_id=sep, n=30, length=64, seed=0)
    assert w.shape == (30, 64)
    assert not (w == sep).any(), "a separator leaked into an analysis window"


def test_window_sampling_is_reproducible():
    toks = np.arange(10_000, dtype=np.uint16) % 100
    a = sample_clean_windows(toks, 999, 10, 32, seed=7)
    b = sample_clean_windows(toks, 999, 10, 32, seed=7)
    assert np.array_equal(a, b)


if __name__ == "__main__":
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
