"""
Tests for the generation metrics (Step 9).

Run:
    python -m common.eval.tests.test_generation
"""

import numpy as np

from common.eval.generation import (rouge_l, distinct_n, repetition_rate, sample_prompts)


# ─────────────────────────────────────────────────────────────────────────────
# ROUGE-L
# ─────────────────────────────────────────────────────────────────────────────

def test_rouge_l_identical_is_one():
    assert abs(rouge_l("a b c d", "a b c d") - 1.0) < 1e-9


def test_rouge_l_disjoint_is_zero():
    assert rouge_l("a b c", "x y z") == 0.0


def test_rouge_l_is_subsequence_not_substring():
    """LCS must allow gaps: 'a c' occurs in 'a b c' as a subsequence."""
    # LCS = 2, precision 2/2, recall 2/3 -> F1 = 0.8
    assert abs(rouge_l("a c", "a b c") - 0.8) < 1e-9


def test_rouge_l_respects_order():
    """A reversed hypothesis shares tokens but not order, so it must score lower."""
    forward = rouge_l("a b c d", "a b c d")
    reversed_ = rouge_l("d c b a", "a b c d")
    assert reversed_ < forward
    # LCS of the reversal is 1, so F1 = 2*(1/4)*(1/4)/(1/2) = 0.25
    assert abs(reversed_ - 0.25) < 1e-9


def test_rouge_l_handles_empty():
    assert rouge_l("", "a b") == 0.0
    assert rouge_l("a b", "") == 0.0


def test_rouge_l_is_symmetric_in_f1():
    """F1 weights precision and recall equally, so swapping arguments must not change it."""
    assert abs(rouge_l("a b c", "a b c d e") - rouge_l("a b c d e", "a b c")) < 1e-9


def test_rouge_l_partial_overlap():
    """Hand-checked: LCS('the cat sat','the cat ran')=2, P=R=2/3, F1=2/3."""
    assert abs(rouge_l("the cat sat", "the cat ran") - 2 / 3) < 1e-9


# ─────────────────────────────────────────────────────────────────────────────
# Diversity and repetition
# ─────────────────────────────────────────────────────────────────────────────

def test_distinct_1_all_unique():
    assert abs(distinct_n(["a b c d"], 1) - 1.0) < 1e-9


def test_distinct_1_all_same():
    """Four identical tokens: 1 unique type out of 4 -> 0.25."""
    assert abs(distinct_n(["a a a a"], 1) - 0.25) < 1e-9


def test_distinct_pools_across_samples():
    """Two samples with identical text must score lower than two different ones."""
    same = distinct_n(["a b c", "a b c"], 1)
    diff = distinct_n(["a b c", "d e f"], 1)
    assert same < diff, "distinct-n is not pooling across samples"
    assert abs(same - 0.5) < 1e-9 and abs(diff - 1.0) < 1e-9


def test_distinct_2_counts_bigrams():
    """'a b a b' has bigrams (a,b),(b,a),(a,b) -> 2 unique of 3."""
    assert abs(distinct_n(["a b a b"], 2) - 2 / 3) < 1e-9


def test_repetition_rate_zero_when_no_repeats():
    assert repetition_rate(["a b c d e f g h"], n=4) == 0.0


def test_repetition_rate_catches_a_loop():
    """The degenerate case this metric exists for: a model stuck in a cycle."""
    loop = " ".join(["a b c d"] * 10)
    assert repetition_rate([loop], n=4) > 0.5, "a repeating loop was not detected"


def test_repetition_rate_averages_per_sample():
    """
    One pathological sample among healthy ones must move the average, but not dominate it
    the way a pooled n-gram count would.
    """
    healthy = " ".join(str(i) for i in range(40))
    loop = " ".join(["a b c d"] * 10)
    mixed = repetition_rate([healthy, healthy, healthy, loop], n=4)
    assert 0.0 < mixed < repetition_rate([loop], n=4)


def test_repetition_rate_skips_too_short():
    """Samples shorter than the n-gram order cannot be judged and must not count."""
    assert repetition_rate(["a b"], n=4) == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Prompt sampling — the correctness issue that would silently skew every score
# ─────────────────────────────────────────────────────────────────────────────

def test_prompts_never_cross_a_document_boundary():
    """
    A window straddling <|endoftext|> would pair a prompt with the opening of an unrelated
    document, so the "reference continuation" is text no model could predict. That would
    depress every overlap metric for a reason unrelated to model quality.
    """
    sep = 7
    toks = np.arange(20_000, dtype=np.uint16) % 100
    toks[::311] = sep                      # scatter separators through the stream
    prompts, refs = sample_prompts(toks, sep_id=sep, n_prompts=40,
                                   prompt_len=16, gen_len=32, seed=0)
    assert len(prompts) == 40
    for p, r in zip(prompts, refs):
        assert sep not in p, "a separator leaked into a prompt"
        assert sep not in r, "a separator leaked into a reference"


def test_prompt_and_reference_are_contiguous():
    """The reference must be exactly the tokens that follow the prompt."""
    toks = np.arange(5_000, dtype=np.uint16) % 60000
    prompts, refs = sample_prompts(toks, sep_id=59_999, n_prompts=10,
                                   prompt_len=8, gen_len=16, seed=1)
    for p, r in zip(prompts, refs):
        joined = p + r
        start = joined[0]
        expected = [(start + i) % 60000 for i in range(len(joined))]
        assert joined == expected, "prompt and reference are not contiguous"


def test_prompt_sampling_is_reproducible():
    toks = np.arange(10_000, dtype=np.uint16) % 100
    a, _ = sample_prompts(toks, 999, 20, 16, 32, seed=3)
    b, _ = sample_prompts(toks, 999, 20, 16, 32, seed=3)
    assert a == b, "prompt sampling is not reproducible under a fixed seed"


def test_prompt_shapes():
    toks = np.arange(10_000, dtype=np.uint16) % 100
    prompts, refs = sample_prompts(toks, 999, 15, 24, 48, seed=0)
    assert all(len(p) == 24 for p in prompts)
    assert all(len(r) == 48 for r in refs)


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
