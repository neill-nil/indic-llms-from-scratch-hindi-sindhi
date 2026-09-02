"""
Tokenize a preprocessed corpus into a flat uint16 binary file.

Training reads token ids, not text. Re-tokenizing 500M tokens every epoch would dominate
the training loop, so each split is encoded once into a flat array of ids that the loader
memory-maps. The result is roughly 1 GB per language rather than the 5 GB of source text.

Why uint16
----------
Both vocabularies are 16,000 entries, comfortably under the 65,536 ceiling of an unsigned
16-bit integer, so every id fits in two bytes. uint32 would double the file size and the
memory bandwidth for no benefit. The writer asserts the vocabulary fits before starting.

Document boundaries
-------------------
Documents are joined by the tokenizer's <|endoftext|> id, inserted directly rather than by
encoding the literal string. Encoding the text "<|endoftext|>" yields *two* tokens — the
SentencePiece word-start marker plus the symbol — which would waste a token at every
boundary and teach the model a spurious two-token separator pattern.

Usage:
    python -m common.data.tokenize_corpus --language hindi
    python -m common.data.tokenize_corpus --language hindi --split val
"""

import argparse
import json
import logging
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np
import sentencepiece as spm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("tokenize_corpus")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DOC_SEPARATOR = "<|endoftext|>"
DTYPE = np.uint16
SPLITS = ("train", "val", "test")

# Lines are encoded in batches so the SentencePiece call overhead is amortised and the
# worker never holds more than this many lines of text at once.
CHUNK_LINES = 20_000

# Set once per worker process by _init_worker; SentencePieceProcessor is not picklable in
# a way that survives being sent through a Pool argument for every task.
_SP = None


def _init_worker(model_path: str):
    global _SP
    _SP = spm.SentencePieceProcessor(model_file=model_path)


def _encode_chunk(args):
    """
    Encode one chunk of items into a flat list of ids.

    An item is either a content line or None, which marks a document boundary. Separators
    travel inside the chunk rather than being handled by the parent, so the parent never
    has to flush the worker pool mid-stream. An earlier version yielded each separator
    separately and forced a full pool drain to keep ordering; with 2.1M Hindi documents
    averaging 3.4 lines each that meant 2.1M synchronisation points on three-line batches,
    which serialised the pool completely and ran ~30x slower than single-threaded.

    All real lines in the chunk are encoded in a single SentencePiece call, then the ids
    are reassembled with separators back in their original positions.
    """
    items, sep_id = args
    lines = [it for it in items if it is not None]
    encoded = iter(_SP.encode(lines)) if lines else iter(())
    out = []
    for item in items:
        if item is None:
            out.append(sep_id)
        else:
            out.extend(next(encoded))
    return out


def iter_item_chunks(path: Path, chunk_size: int = CHUNK_LINES):
    """
    Stream the split file as chunks of items, where an item is a content line or None for
    a document boundary. Chunk size is bounded so a worker never holds more than this many
    items, keeping memory flat regardless of corpus size.
    """
    batch = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped == DOC_SEPARATOR:
                batch.append(None)
            elif stripped:
                batch.append(stripped)
            else:
                continue
            if len(batch) >= chunk_size:
                yield batch
                batch = []
    if batch:
        yield batch


def tokenize_split(language: str, split: str, model_path: Path, workers: int) -> dict:
    """
    Encode one split to <split>.bin and return its statistics.

    Chunks are dispatched to a worker pool and collected strictly in order, with the queue
    of outstanding jobs bounded so peak memory stays flat on a multi-GB corpus.
    """
    src = PROJECT_ROOT / language / "data" / "preprocessed" / f"{split}.txt"
    out_dir = PROJECT_ROOT / language / "data" / "tokenized"
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / f"{split}.bin"

    if not src.exists():
        raise FileNotFoundError(src)

    sp = spm.SentencePieceProcessor(model_file=str(model_path))
    vocab_size = sp.get_piece_size()
    if vocab_size > np.iinfo(DTYPE).max + 1:
        raise ValueError(
            f"vocabulary of {vocab_size:,} does not fit in {DTYPE.__name__}; "
            f"use a wider dtype")
    sep_id = sp.piece_to_id(DOC_SEPARATOR)
    if sep_id == sp.unk_id():
        raise ValueError(f"{DOC_SEPARATOR} is not a token in {model_path.name}")

    logger.info("Tokenizing %s/%s -> %s (sep id %d, %d workers)",
                language, split, dst.name, sep_id, workers)

    start = time.time()
    total = docs = max_id = 0
    last_log = start

    with open(dst, "wb") as f_out, \
         mp.Pool(workers, initializer=_init_worker, initargs=(str(model_path),)) as pool:

        pending = []

        def write_one():
            """Pop the oldest job, in dispatch order, and append its ids to the file."""
            nonlocal total, max_id
            arr = np.asarray(pending.pop(0).get(), dtype=DTYPE)
            arr.tofile(f_out)
            total += arr.size
            if arr.size:
                max_id = max(max_id, int(arr.max()))

        for chunk in iter_item_chunks(src):
            docs += sum(1 for item in chunk if item is None)
            pending.append(pool.apply_async(_encode_chunk, ((chunk, sep_id),)))
            # Keep a couple of jobs per worker in flight: enough to hide latency, few
            # enough that memory stays bounded.
            while len(pending) > 2 * workers:
                write_one()
            if time.time() - last_log > 30:
                logger.info("  ...%s tokens, %s documents, %.0f tok/s",
                            f"{total:,}", f"{docs:,}", total / (time.time() - start))
                last_log = time.time()

        while pending:
            write_one()

    elapsed = time.time() - start
    size = dst.stat().st_size
    assert size == total * np.dtype(DTYPE).itemsize, "written bytes do not match token count"
    if max_id >= vocab_size:
        raise ValueError(f"token id {max_id} exceeds vocabulary size {vocab_size}")

    logger.info("  %s: %s tokens, %s documents, %.2f GB, %.1fs (%s tok/s)",
                dst.name, f"{total:,}", f"{docs:,}", size / 1e9, elapsed,
                f"{int(total / max(elapsed, 1e-9)):,}")

    return {"split": split, "tokens": total, "documents": docs, "bytes": size,
            "max_token_id": max_id, "seconds": round(elapsed, 1)}


def main():
    parser = argparse.ArgumentParser(description="Tokenize a corpus to a uint16 .bin")
    parser.add_argument("--language", required=True, choices=["hindi", "sindhi"])
    parser.add_argument("--split", choices=SPLITS, help="default: all three")
    parser.add_argument("--vocab_size", type=int, default=16000)
    parser.add_argument("--workers", type=int,
                        default=max((os.cpu_count() or 4) - 2, 2))
    args = parser.parse_args()

    model_path = (PROJECT_ROOT / args.language / "tokenizer"
                  / f"{args.language}_unigram_{args.vocab_size}.model")
    if not model_path.exists():
        raise FileNotFoundError(model_path)

    splits = [args.split] if args.split else list(SPLITS)
    stats = {"language": args.language, "tokenizer": model_path.name,
             "dtype": DTYPE.__name__, "splits": {}}
    for split in splits:
        stats["splits"][split] = tokenize_split(args.language, split, model_path, args.workers)

    out = PROJECT_ROOT / args.language / "data" / "tokenized" / "tokenization_stats.json"
    # Merge with any previous run so tokenizing one split does not discard the others.
    if out.exists():
        previous = json.loads(out.read_text())
        if previous.get("tokenizer") == stats["tokenizer"]:
            merged = previous.get("splits", {})
            merged.update(stats["splits"])
            stats["splits"] = merged
    out.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    logger.info("Wrote %s", out)


if __name__ == "__main__":
    main()
