"""
Tests for the training loop (Step 6).

The important one is test_resume_reproduces_uninterrupted_run: it trains N steps in one
go, then trains the same N steps with a kill and reload in the middle, and asserts the
resulting weights are bitwise identical. The brief warns that a run which cannot resume
loses marks, so this is verified rather than assumed.

Run:
    python -m common.train.tests.test_trainer
"""

import logging
import math
import tempfile
from pathlib import Path

import numpy as np
import torch

from common.data.loader import TokenDataset
from common.model.config import GPTConfig
from common.model.gpt import GPT
from common.train.trainer import TrainConfig, Trainer, select_precision

logging.getLogger("trainer").setLevel(logging.WARNING)


def _toy(tmp: Path, n=60_000, vocab=97):
    """A small deterministic corpus with real structure, so loss can actually fall."""
    rng = np.random.default_rng(0)
    # A repeating pattern plus noise: learnable, unlike uniform random ids.
    base = np.arange(n) % 40
    noise = rng.integers(0, vocab, n)
    toks = np.where(rng.random(n) < 0.7, base, noise).astype(np.uint16)
    for split in ("train", "val"):
        toks.tofile(tmp / f"{split}.bin")
    return vocab


def _setup(tmp: Path, **train_kw):
    vocab = _toy(tmp)
    cfg = GPTConfig(vocab_size=vocab, d_model=32, n_heads=4, d_ff=64,
                    max_seq_len=16, n_layers=2, dropout=0.0, language="test")
    tcfg = TrainConfig(**{"max_tokens": 16 * 2 * 20, "micro_batch": 2, "grad_accum": 2,
                          "warmup_steps": 2, "eval_every": 10_000,
                          "checkpoint_every": 10_000, "log_every": 10_000, **train_kw})
    torch.manual_seed(0)
    model = GPT(cfg)
    train = TokenDataset(tmp / "train.bin", cfg.max_seq_len)
    val = TokenDataset(tmp / "val.bin", cfg.max_seq_len)
    return model, train, val, tcfg


def _weights(model):
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


# ─────────────────────────────────────────────────────────────────────────────
# Resume — the property the whole checkpoint format exists for
# ─────────────────────────────────────────────────────────────────────────────

def test_resume_reproduces_uninterrupted_run():
    """
    Twenty steps straight through must equal ten steps, a save, a fresh process-equivalent
    reload, and ten more. Bitwise, not approximately: any drift means optimizer moments,
    RNG state or the data cursor were not fully restored, and a resumed run would silently
    diverge from the one it claims to continue.
    """
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)

        # Reference: uninterrupted.
        model, train, val, tcfg = _setup(tmp)
        ref = Trainer(model, train, val, tcfg, tmp / "ref", device="cpu")
        ref.train()
        reference = _weights(ref.model)

        # Interrupted: stop halfway, save, rebuild everything, resume. The config is
        # identical to the reference run — only the stopping point differs, which is what
        # an interruption actually looks like. Halving max_tokens instead would compress
        # the cosine schedule and change the learning rates, so the two runs would differ
        # for a reason that has nothing to do with checkpointing.
        model2, train2, val2, tcfg2 = _setup(tmp)
        a = Trainer(model2, train2, val2, tcfg2, tmp / "run", device="cpu")
        a.train(until_step=a.total_steps // 2)
        a.save_checkpoint("latest")
        assert 0 < a.step < a.total_steps, "the interruption did not land mid-run"
        del a, model2

        # Fresh objects, as after a process restart.
        model3, train3, val3, tcfg3 = _setup(tmp)
        b = Trainer(model3, train3, val3, tcfg3, tmp / "run", device="cpu")
        b.load_checkpoint(tmp / "run" / "latest.pt")
        b.train()
        resumed = _weights(b.model)

        assert b.step == ref.step, f"step {b.step} != {ref.step}"
        assert b.tokens_seen == ref.tokens_seen
        for k in reference:
            assert torch.equal(reference[k], resumed[k]), (
                f"{k} differs after resume; max delta "
                f"{(reference[k] - resumed[k]).abs().max().item():.3e}")


def test_resume_works_on_gpu():
    """
    Resume on CUDA specifically.

    The CPU test cannot catch device-related restore bugs: torch.load(map_location="cuda")
    moves the saved RNG ByteTensors onto the GPU, and Generator.set_state rejects
    non-CPU tensors. That failure only appears on the real training path.
    """
    if not torch.cuda.is_available():
        print("      (skipped: no CUDA)")
        return
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        model, train, val, tcfg = _setup(tmp)
        ref = Trainer(model, train, val, tcfg, tmp / "ref", device="cuda")
        ref.train()
        reference = _weights(ref.model)

        model2, train2, val2, tcfg2 = _setup(tmp)
        a = Trainer(model2, train2, val2, tcfg2, tmp / "run", device="cuda")
        a.train(until_step=a.total_steps // 2)
        a.save_checkpoint("latest")
        del a, model2

        model3, train3, val3, tcfg3 = _setup(tmp)
        b = Trainer(model3, train3, val3, tcfg3, tmp / "run", device="cuda")
        b.load_checkpoint(tmp / "run" / "latest.pt")
        b.train()

        assert b.step == ref.step
        for k in reference:
            assert torch.equal(reference[k], _weights(b.model)[k]), \
                f"{k} differs after a CUDA resume"


def test_checkpoint_contains_everything_needed():
    """The brief requires weights, optimizer, scheduler position, step and config."""
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        model, train, val, tcfg = _setup(tmp)
        t = Trainer(model, train, val, tcfg, tmp / "run", device="cpu")
        ckpt = torch.load(t.save_checkpoint("latest"), map_location="cpu",
                          weights_only=False)
        for key in ["model", "optimizer", "scaler", "step", "tokens_seen",
                    "model_config", "train_config", "data_gen_state",
                    "torch_rng_state", "best_val_loss"]:
            assert key in ckpt, f"checkpoint is missing {key}"
        # The scheduler is a pure function of step, so the step counter is the scheduler
        # state; assert the LR is reconstructible from it.
        assert math.isclose(t.lr_at(ckpt["step"]), t.lr_at(t.step))


def test_checkpoint_write_is_atomic():
    """A checkpoint must never be left half-written under the final name."""
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        model, train, val, tcfg = _setup(tmp)
        t = Trainer(model, train, val, tcfg, tmp / "run", device="cpu")
        path = t.save_checkpoint("latest")
        assert path.exists()
        assert not path.with_suffix(".pt.tmp").exists(), "temp file was left behind"


def test_resume_rejects_mismatched_architecture():
    """Loading a checkpoint into a differently-shaped model must fail loudly."""
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        model, train, val, tcfg = _setup(tmp)
        t = Trainer(model, train, val, tcfg, tmp / "run", device="cpu")
        path = t.save_checkpoint("latest")

        other = GPTConfig(vocab_size=97, d_model=32, n_heads=4, d_ff=64,
                          max_seq_len=16, n_layers=4, dropout=0.0, language="test")
        t2 = Trainer(GPT(other), train, val, tcfg, tmp / "run2", device="cpu")
        try:
            t2.load_checkpoint(path)
        except ValueError as e:
            assert "does not match" in str(e)
        else:
            raise AssertionError("expected a ValueError on architecture mismatch")


# ─────────────────────────────────────────────────────────────────────────────
# Optimisation behaviour
# ─────────────────────────────────────────────────────────────────────────────

def test_loss_decreases_on_learnable_data():
    """A smoke test that the loop actually optimises."""
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        model, train, val, tcfg = _setup(tmp, max_tokens=16 * 2 * 2 * 150,
                                         learning_rate=3e-3, warmup_steps=5)
        t = Trainer(model, train, val, tcfg, tmp / "run", device="cpu")
        before = t.evaluate()
        t.train()
        after = t.evaluate()
        assert after < before - 0.05, f"loss did not fall: {before:.4f} -> {after:.4f}"


def test_lr_schedule_shape():
    """Warmup rises linearly from ~0 to peak, then cosine-decays to min_lr."""
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        model, train, val, tcfg = _setup(tmp, warmup_steps=10)
        t = Trainer(model, train, val, tcfg, tmp / "run", device="cpu")

        assert t.lr_at(0) < t.cfg.learning_rate, "no warmup at step 0"
        assert math.isclose(t.lr_at(9), t.cfg.learning_rate, rel_tol=1e-6), "peak not hit"
        warm = [t.lr_at(i) for i in range(10)]
        assert warm == sorted(warm), "warmup is not monotonically increasing"

        after = [t.lr_at(i) for i in range(10, t.total_steps)]
        assert after == sorted(after, reverse=True), "post-warmup LR is not decaying"
        assert t.lr_at(t.total_steps) >= t.cfg.min_lr - 1e-9, "LR fell below min_lr"


def test_weight_decay_excludes_biases_and_norms():
    """One-dimensional parameters must sit in the no-decay group."""
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        model, train, val, tcfg = _setup(tmp)
        t = Trainer(model, train, val, tcfg, tmp / "run", device="cpu")
        decay, no_decay = t.optimizer.param_groups
        assert decay["weight_decay"] == tcfg.weight_decay
        assert no_decay["weight_decay"] == 0.0
        assert all(p.dim() >= 2 for p in decay["params"])
        assert all(p.dim() < 2 for p in no_decay["params"])


def test_grad_accum_matches_large_batch():
    """
    Accumulating G micro-batches must give the same gradient as one batch G times larger.
    If the per-micro-batch loss were not divided by G, the effective learning rate would
    silently scale with the accumulation factor.
    """
    torch.manual_seed(0)
    cfg = GPTConfig(vocab_size=50, d_model=32, n_heads=4, d_ff=64,
                    max_seq_len=8, n_layers=1, dropout=0.0)
    model = GPT(cfg)
    x = torch.randint(0, 50, (8, 8))
    y = torch.randint(0, 50, (8, 8))

    model.zero_grad()
    _, loss = model(x, y)
    loss.backward()
    big = {k: v.grad.clone() for k, v in model.named_parameters()}

    model.zero_grad()
    for i in range(4):
        _, l = model(x[i * 2:(i + 1) * 2], y[i * 2:(i + 1) * 2])
        (l / 4).backward()
    accum = {k: v.grad.clone() for k, v in model.named_parameters()}

    for k in big:
        assert torch.allclose(big[k], accum[k], atol=1e-5), \
            f"{k}: accumulated gradient differs from the equivalent large batch"


def test_evaluate_is_deterministic():
    """Eval must score the same windows every call, or improvements vanish into noise."""
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        model, train, val, tcfg = _setup(tmp, eval_batches=5)
        t = Trainer(model, train, val, tcfg, tmp / "run", device="cpu")
        assert math.isclose(t.evaluate(), t.evaluate(), rel_tol=1e-9)


def test_evaluate_restores_training_mode():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        model, train, val, tcfg = _setup(tmp, eval_batches=2)
        t = Trainer(model, train, val, tcfg, tmp / "run", device="cpu")
        t.model.train()
        t.evaluate()
        assert t.model.training, "evaluate() left the model in eval mode"


def test_precision_selection():
    """bf16 where supported, fp16 + scaler otherwise, fp32 on CPU."""
    dtype, scaler = select_precision("cpu")
    assert dtype == torch.float32 and not scaler
    if torch.cuda.is_available():
        dtype, scaler = select_precision("cuda")
        if torch.cuda.is_bf16_supported():
            assert dtype == torch.bfloat16 and not scaler
        else:
            assert dtype == torch.float16 and scaler, \
                "pre-Ampere GPUs need fp16 with a gradient scaler"


def test_training_log_is_written():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        model, train, val, tcfg = _setup(tmp, log_every=1, max_tokens=16 * 2 * 2 * 5)
        t = Trainer(model, train, val, tcfg, tmp / "run", device="cpu")
        t.train()
        lines = t.log_path.read_text().strip().splitlines()
        assert lines, "no training log written"
        import json
        rec = json.loads(lines[0])
        assert {"step", "tokens", "train_loss", "lr"} <= set(rec)


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
