"""
Tests for the tokenization and batching pipeline (Step 5).

These run against a small synthetic corpus written to a temp directory, so they do not
depend on the multi-GB real corpora being tokenized first.

Run:
    python -m common.data.tests.test_data
"""

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

from common.data.loader import TokenDataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DOC_SEPARATOR = "<|endoftext|>"


def _write_bin(tmp: Path, tokens) -> Path:
    p = tmp / "toy.bin"
    np.asarray(tokens, dtype=np.uint16).tofile(p)
    return p


# ─────────────────────────────────────────────────────────────────────────────
# Loader: the input/target shift is the thing most worth getting right
# ─────────────────────────────────────────────────────────────────────────────

def test_targets_are_inputs_shifted_by_one():
    """
    y[:, t] must be the token that follows x[:, t]. An off-by-one here trains the model to
    predict the current token instead of the next one, which still produces a falling loss
    curve and is therefore easy to miss.
    """
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        ds = TokenDataset(_write_bin(tmp, range(200)), block_size=8)
        g = torch.Generator().manual_seed(0)
        x, y = ds.get_batch(4, generator=g)
        assert torch.equal(y[:, :-1], x[:, 1:]), "targets are not inputs shifted by one"
        # And the shift is exactly one against the underlying stream.
        for row in range(x.size(0)):
            start = int(x[row, 0].item())
            assert torch.equal(x[row], torch.arange(start, start + 8))
            assert torch.equal(y[row], torch.arange(start + 1, start + 9))


def test_batch_shapes_and_dtype():
    with tempfile.TemporaryDirectory() as d:
        ds = TokenDataset(_write_bin(Path(d), range(500)), block_size=16)
        x, y = ds.get_batch(6)
        assert x.shape == y.shape == (6, 16)
        assert x.dtype == torch.int64, "embedding lookup requires int64"


def test_never_reads_past_the_end():
    """
    The last legal start offset must leave room for block_size inputs plus the shifted
    target. Exhaustively check every offset the dataset claims is valid.
    """
    with tempfile.TemporaryDirectory() as d:
        n, block = 64, 8
        ds = TokenDataset(_write_bin(Path(d), range(n)), block_size=block)
        assert len(ds) == n - block
        for s in range(len(ds)):
            assert s + block + 1 <= n, f"offset {s} would read past the end"
        # Drawing many batches must never raise or produce a short row.
        for _ in range(50):
            x, y = ds.get_batch(8)
            assert x.shape[1] == block and y.shape[1] == block


def test_rejects_block_size_larger_than_corpus():
    with tempfile.TemporaryDirectory() as d:
        try:
            TokenDataset(_write_bin(Path(d), range(10)), block_size=32)
        except ValueError as e:
            assert "too few" in str(e)
        else:
            raise AssertionError("expected ValueError for an oversized block")


def test_batches_are_reproducible_with_a_seeded_generator():
    """Needed so a resumed run can reproduce the data order."""
    with tempfile.TemporaryDirectory() as d:
        ds = TokenDataset(_write_bin(Path(d), range(1000)), block_size=8)
        a = ds.get_batch(4, generator=torch.Generator().manual_seed(123))[0]
        b = ds.get_batch(4, generator=torch.Generator().manual_seed(123))[0]
        assert torch.equal(a, b)


def test_sequential_iteration_covers_every_token_once():
    """Evaluation must score the split deterministically, without overlap or gaps."""
    with tempfile.TemporaryDirectory() as d:
        n, block = 100, 10
        ds = TokenDataset(_write_bin(Path(d), range(n)), block_size=block)
        seen = []
        for x, y in ds.iter_sequential(batch_size=3):
            assert x.shape[1] == block
            seen.extend(x.flatten().tolist())
        # Windows are non-overlapping and stride by block_size.
        assert len(seen) == len(set(seen)), "sequential windows overlap"
        assert seen == sorted(seen), "sequential windows are out of order"


def test_missing_file_message_is_actionable():
    try:
        TokenDataset(Path("/nonexistent/train.bin"), block_size=8)
    except FileNotFoundError as e:
        assert "tokenize_corpus" in str(e), "error should say how to fix it"
    else:
        raise AssertionError("expected FileNotFoundError")


# ─────────────────────────────────────────────────────────────────────────────
# Round trip through the real tokenizer
# ─────────────────────────────────────────────────────────────────────────────

def test_end_to_end_roundtrip_with_real_tokenizer():
    """
    Build a small corpus in the real on-disk format, run the real tokenizer script over
    it, and confirm the resulting .bin decodes back to the original text with document
    separators in the right places.
    """
    import sentencepiece as spm

    model = PROJECT_ROOT / "hindi" / "tokenizer" / "hindi_unigram_16000.model"
    if not model.exists():
        print("      (skipped: Hindi tokenizer not present)")
        return

    sp = spm.SentencePieceProcessor(model_file=str(model))
    sep_id = sp.piece_to_id(DOC_SEPARATOR)
    docs = ["यह पहला वाक्य है।", "यह दूसरा दस्तावेज़ है।", "तीसरा।"]

    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        # Mimic the corpus layout the pipeline produces.
        lang_dir = tmp / "hindi" / "data" / "preprocessed"
        lang_dir.mkdir(parents=True)
        (lang_dir / "val.txt").write_text(
            "".join(f"{doc}\n{DOC_SEPARATOR}\n" for doc in docs), encoding="utf-8")
        (tmp / "hindi" / "tokenizer").mkdir(parents=True)
        for suffix in (".model", ".vocab"):
            src = model.with_suffix(suffix)
            (tmp / "hindi" / "tokenizer" / src.name).write_bytes(src.read_bytes())

        env = {"PYTHONPATH": str(PROJECT_ROOT)}
        import os
        env = {**os.environ, **env}
        # Patch PROJECT_ROOT by running the module with cwd at the temp tree is not
        # possible, so drive tokenize_split directly against the temp layout instead.
        import common.data.tokenize_corpus as tc
        original_root = tc.PROJECT_ROOT
        tc.PROJECT_ROOT = tmp
        try:
            stats = tc.tokenize_split("hindi", "val",
                                      tmp / "hindi" / "tokenizer" / model.name, workers=2)
        finally:
            tc.PROJECT_ROOT = original_root

        arr = np.fromfile(tmp / "hindi" / "data" / "tokenized" / "val.bin", dtype=np.uint16)
        assert stats["documents"] == 3, stats
        assert stats["tokens"] == arr.size

        # The separator must appear exactly once per document, as a single id.
        assert int((arr == sep_id).sum()) == 3, "wrong number of document separators"

        # Split on the separator and decode each segment back to its source text.
        segments, current = [], []
        for tok in arr.tolist():
            if tok == sep_id:
                segments.append(current); current = []
            else:
                current.append(tok)
        assert len(segments) == 3
        for original, ids in zip(docs, segments):
            assert sp.decode(ids) == original, f"{sp.decode(ids)!r} != {original!r}"


def test_separator_is_one_token_not_two():
    """
    Encoding the literal string costs two tokens (word-start marker + symbol). The writer
    must inject the raw id instead.
    """
    import sentencepiece as spm
    model = PROJECT_ROOT / "hindi" / "tokenizer" / "hindi_unigram_16000.model"
    if not model.exists():
        print("      (skipped: Hindi tokenizer not present)")
        return
    sp = spm.SentencePieceProcessor(model_file=str(model))
    assert len(sp.encode(DOC_SEPARATOR)) == 2, \
        "premise changed: encoding the separator no longer costs two tokens"
    assert sp.piece_to_id(DOC_SEPARATOR) != sp.unk_id()


def test_uint16_is_wide_enough_for_both_vocabularies():
    import sentencepiece as spm
    for lang in ("hindi", "sindhi"):
        model = PROJECT_ROOT / lang / "tokenizer" / f"{lang}_unigram_16000.model"
        if not model.exists():
            continue
        v = spm.SentencePieceProcessor(model_file=str(model)).get_piece_size()
        assert v <= np.iinfo(np.uint16).max + 1, f"{lang}: vocab {v} overflows uint16"


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
