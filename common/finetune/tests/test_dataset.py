"""
Tests for the reasoning finetuning dataset and its loss masking.

The masking is the part most able to fail silently. If the prompt were left unmasked the
model would still train, the loss would still fall, and the only symptom would be a model
that learned the task less well than it should have. These tests assert the properties the
design claims rather than merely exercising the code.

    python -m common.finetune.tests.test_dataset
"""

import sys
from pathlib import Path

import torch

from common.finetune.dataset import (IGNORE_INDEX, MAX_LEN, DOC_SEPARATOR,
                                     load_reasoning_split)
from common.model.config import GPTConfig
from common.model.gpt import GPT

PASSED, FAILED = [], []


def check(name, fn):
    try:
        fn()
        PASSED.append(name)
        print(f"  PASS  {name}")
    except Exception as exc:                                   # noqa: BLE001
        FAILED.append((name, exc))
        print(f"  FAIL  {name}: {exc}")


DS = load_reasoning_split("hindi", "val", limit=300)


def test_only_answer_tokens_are_supervised():
    """Every supervised target must decode to the answer, and nothing else."""
    for row in range(50):
        ex = DS.examples[row]
        sup = [int(t) for t in DS.y[row] if t != IGNORE_INDEX]
        decoded = DS.sp.decode(sup)
        assert decoded.endswith(DOC_SEPARATOR), f"row {row}: answer must end with eos"
        assert DS.sp.decode(sup[:-1]) == ex["answer"], (
            f"row {row}: supervised span {DS.sp.decode(sup[:-1])!r} "
            f"!= answer {ex['answer']!r}")


def test_prompt_positions_are_masked():
    """No position before the answer may carry a target."""
    for row in range(50):
        first = DS.answer_starts[row]
        assert (DS.y[row, :first] == IGNORE_INDEX).all(), \
            f"row {row}: a prompt position is supervised"
        assert DS.y[row, first] != IGNORE_INDEX, \
            f"row {row}: the first answer position is masked"


def test_padding_is_masked_and_never_a_target():
    """Padding must be invisible to the loss."""
    for row in range(50):
        pad_cols = DS.x[row] == DS.pad_id
        # Every padded input column past the real sequence must have a masked target.
        tail = DS.y[row][pad_cols]
        assert (tail == IGNORE_INDEX).all(), f"row {row}: a pad position is supervised"


def test_no_example_is_truncated_at_the_default_length():
    """MAX_LEN was chosen from measurement; assert the measurement still holds."""
    for split in ("train", "val", "test_iid", "test_entity", "test_template", "test_hop"):
        for language in ("hindi", "sindhi"):
            ds = load_reasoning_split(language, split, limit=500)
            for row, ex in enumerate(ds.examples):
                full = (len(ds.sp.encode(ex["prompt"]))
                        + len(ds.sp.encode(" " + ex["answer"])) + 1)
                assert full <= MAX_LEN + 1, (
                    f"{language}/{split} row {row}: {full} tokens exceeds "
                    f"MAX_LEN={MAX_LEN}; raise it or the answer will be truncated")


def test_right_padding_cannot_influence_a_supervised_position():
    """
    The claim that right padding needs no attention mask, tested rather than argued.

    Corrupting every padded input column must leave the logits at all supervised
    positions bitwise identical. If it does not, padding is leaking into the loss and the
    attention implementation would need a padding mask after all.
    """
    torch.manual_seed(0)
    cfg = GPTConfig(vocab_size=16000, d_model=64, n_layers=2, n_heads=4, d_ff=128,
                    max_seq_len=MAX_LEN, dropout=0.0)
    model = GPT(cfg).eval()

    x, y = DS._gather(list(range(8)), "cpu")
    with torch.no_grad():
        clean, _ = model(x)
        corrupted_x = x.clone()
        pad_cols = x == DS.pad_id
        corrupted_x[pad_cols] = torch.randint(
            0, cfg.vocab_size, (int(pad_cols.sum()),), dtype=torch.long)
        dirty, _ = model(corrupted_x)

    supervised = y != IGNORE_INDEX
    assert torch.equal(clean[supervised], dirty[supervised]), (
        "changing pad tokens changed the logits at a supervised position — right padding "
        "is NOT safe and the attention needs a padding mask")
    # And the test must not pass vacuously: the padded positions themselves did change.
    assert not torch.equal(clean[pad_cols], dirty[pad_cols]), \
        "corruption had no effect anywhere; the test is not exercising anything"


def test_get_batch_matches_token_dataset_contract():
    """Shapes and dtypes must match what Trainer expects from TokenDataset."""
    gen = torch.Generator().manual_seed(0)
    x, y = DS.get_batch(4, "cpu", generator=gen)
    assert x.shape == y.shape == (4, MAX_LEN), f"got {x.shape}, {y.shape}"
    assert x.dtype == y.dtype == torch.int64
    # Same generator state must reproduce the same batch.
    gen2 = torch.Generator().manual_seed(0)
    x2, _ = DS.get_batch(4, "cpu", generator=gen2)
    assert torch.equal(x, x2), "batching is not reproducible from the generator seed"


def test_sequential_iteration_covers_every_example_once():
    seen = sum(xb.size(0) for xb, _ in DS.iter_sequential(16, "cpu"))
    assert seen == len(DS), f"{seen} != {len(DS)}"


if __name__ == "__main__":
    print("finetune dataset tests")
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            check(name, fn)
    print(f"\n{len(PASSED)}/{len(PASSED) + len(FAILED)} passed")
    sys.exit(1 if FAILED else 0)
