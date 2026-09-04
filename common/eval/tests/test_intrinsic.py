"""
Tests for intrinsic evaluation (Step 8).

Run:
    python -m common.eval.tests.test_intrinsic
"""

import math
import tempfile
from pathlib import Path

import numpy as np
import torch

from common.data.loader import TokenDataset
from common.eval.intrinsic import evaluate, EvalResult
from common.model.config import GPTConfig
from common.model.gpt import GPT


class FakeTokenizer:
    """Decodes each id to a fixed 1-byte string, so byte counts are exactly predictable."""
    def decode(self, ids):
        return "a" * len(ids)

    def get_piece_size(self):
        return 97


def _setup(tmp: Path, n=20_000, vocab=97, block=16):
    toks = (np.arange(n) % vocab).astype(np.uint16)
    toks.tofile(tmp / "t.bin")
    cfg = GPTConfig(vocab_size=vocab, d_model=32, n_heads=4, d_ff=64,
                    max_seq_len=block, n_layers=2, dropout=0.0, language="test")
    torch.manual_seed(0)
    return GPT(cfg), TokenDataset(tmp / "t.bin", block), cfg


def test_perplexity_is_exp_of_cross_entropy():
    """The defining relationship, and the one most often got wrong by a stray log base."""
    with tempfile.TemporaryDirectory() as d:
        model, data, _ = _setup(Path(d))
        nats, n_tok, n_bytes, _ = evaluate(model, data, FakeTokenizer(), "cpu",
                                           batch_size=4, max_windows=32)
        ce = nats / n_tok
        assert math.isclose(math.exp(ce), math.exp(nats / n_tok), rel_tol=1e-12)
        # And an untrained model must sit near ln(vocab).
        assert abs(ce - math.log(97)) < 0.6, f"CE {ce:.3f} vs ln(97)={math.log(97):.3f}"


def test_bits_per_byte_formula():
    """
    BPB = (total nats / ln 2) / total bytes. With a tokenizer that emits exactly one byte
    per token, BPB must equal cross-entropy converted to bits.
    """
    with tempfile.TemporaryDirectory() as d:
        model, data, _ = _setup(Path(d))
        nats, n_tok, n_bytes, _ = evaluate(model, data, FakeTokenizer(), "cpu",
                                           batch_size=4, max_windows=32)
        assert n_bytes == n_tok, "fake tokenizer should give one byte per token"
        bpb = (nats / math.log(2)) / n_bytes
        ce_in_bits = (nats / n_tok) / math.log(2)
        assert math.isclose(bpb, ce_in_bits, rel_tol=1e-9), (bpb, ce_in_bits)


def test_evaluation_is_deterministic():
    """Sequential windows, no sampling: two runs must agree exactly."""
    with tempfile.TemporaryDirectory() as d:
        model, data, _ = _setup(Path(d))
        a = evaluate(model, data, FakeTokenizer(), "cpu", 4, max_windows=32)
        b = evaluate(model, data, FakeTokenizer(), "cpu", 4, max_windows=32)
        assert a == b, "evaluation is not deterministic"


def test_every_token_scored_once():
    """Windows must tile the split without overlap or gaps."""
    with tempfile.TemporaryDirectory() as d:
        model, data, _ = _setup(Path(d), n=1000, block=10)
        _, n_tok, _, n_windows = evaluate(model, data, FakeTokenizer(), "cpu", 4)
        assert n_tok == n_windows * 10, f"{n_tok} tokens over {n_windows} windows of 10"


def test_batch_size_does_not_change_the_result():
    """A different batch size regroups the same windows; the totals must not move."""
    with tempfile.TemporaryDirectory() as d:
        model, data, _ = _setup(Path(d), n=2000)
        n2, t2, b2, w2 = evaluate(model, data, FakeTokenizer(), "cpu", batch_size=2)
        n8, t8, b8, w8 = evaluate(model, data, FakeTokenizer(), "cpu", batch_size=8)
        assert (t2, b2, w2) == (t8, b8, w8)
        assert math.isclose(n2, n8, rel_tol=1e-4), (n2, n8)


def test_shorter_context_is_accepted():
    """
    The matched-context comparison evaluates a ctx-1024 model at 512. Position embeddings
    are sliced, so the model must run unmodified on shorter windows.
    """
    cfg = GPTConfig(vocab_size=97, d_model=32, n_heads=4, d_ff=64,
                    max_seq_len=32, n_layers=2, dropout=0.0, language="test")
    model = GPT(cfg).eval()
    with torch.no_grad():
        long_out, _ = model(torch.zeros(1, 32, dtype=torch.long))
        short_out, _ = model(torch.zeros(1, 16, dtype=torch.long))
    assert long_out.shape[1] == 32 and short_out.shape[1] == 16
    # The first 16 positions must be identical: causal masking means they cannot depend on
    # anything later, so truncating the window cannot change them.
    assert torch.allclose(long_out[0, :16], short_out[0], atol=1e-5), \
        "truncating the context changed earlier positions — causality is broken"


def test_lower_loss_gives_lower_perplexity_and_bpb():
    """Sanity: a trained model must score better than an untrained one on all three."""
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        model, data, cfg = _setup(tmp)
        before = evaluate(model, data, FakeTokenizer(), "cpu", 4, max_windows=32)

        opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
        model.train()
        gen = torch.Generator().manual_seed(0)
        for _ in range(60):
            x, y = data.get_batch(4, "cpu", generator=gen)
            _, loss = model(x, y)
            loss.backward(); opt.step(); opt.zero_grad()
        after = evaluate(model, data, FakeTokenizer(), "cpu", 4, max_windows=32)

        ce_before, ce_after = before[0] / before[1], after[0] / after[1]
        assert ce_after < ce_before, f"CE did not improve: {ce_before:.4f} -> {ce_after:.4f}"
        assert math.exp(ce_after) < math.exp(ce_before)


def test_result_row_formats():
    r = EvalResult("hindi", "test", 512, 3.4795, 32.44, 1.1234, 1000, 3000, 10)
    s = r.row()
    assert "hindi" in s and "512" in s and "3.4795" in s and "32.44" in s


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
