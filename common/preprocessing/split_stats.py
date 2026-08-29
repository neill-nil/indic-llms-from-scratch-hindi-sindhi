"""
Measure the actual train/validation/test splits on disk and write split_statistics.json.

This exists because the split statistics quoted in the Phase 1 report were inherited from
an earlier two-way (train/val) revision of the pipeline and no longer described the files
that were actually shipped: both eval splits were quoted at equal size when the MinHash
pass had in fact pruned them by different amounts, and the "total tokens" figure was in
reality the train split alone.

Everything here is measured from the split files themselves, so the numbers cannot drift
away from the artefacts again.

Usage:
    python -m common.preprocessing.split_stats --language hindi
"""

import argparse
import json
import logging
from pathlib import Path

import sentencepiece as spm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("split_stats")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DOC_SEPARATOR = "<|endoftext|>"
SPLITS = ("train", "val", "test")


def measure_split(path: Path, sp: spm.SentencePieceProcessor, token_stride: int = 1):
    """
    Count documents, lines, bytes and tokens in one split file.

    `token_stride` subsamples lines for the token count on very large files: encoding a
    5 GB train split in full is slow, so we encode every Nth line and scale up. Documents,
    lines and bytes are always counted exactly.
    """
    docs = lines = 0
    total_bytes = path.stat().st_size
    sampled_tokens = sampled_lines = 0

    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            stripped = line.strip()
            if stripped == DOC_SEPARATOR:
                docs += 1
                continue
            if not stripped:
                continue
            lines += 1
            if i % token_stride == 0:
                sampled_tokens += len(sp.encode(stripped))
                sampled_lines += 1

    # Scale the sampled token count back up by the sampling rate, then add one token per
    # document boundary. Phase 2 injects the <|endoftext|> id directly into the token
    # stream rather than encoding the literal string, which would cost two tokens (the
    # SentencePiece word-start marker plus the symbol itself).
    tokens = int(sampled_tokens * (lines / sampled_lines)) if sampled_lines else 0
    tokens += docs

    return {
        "documents": docs,
        "content_lines": lines,
        "bytes": total_bytes,
        "megabytes": round(total_bytes / 1024 / 1024, 1),
        "tokens": tokens,
        "tokens_exact": token_stride == 1,
        "token_sample_stride": token_stride,
    }


def main():
    parser = argparse.ArgumentParser(description="Measure actual dataset splits on disk")
    parser.add_argument("--language", required=True, choices=["hindi", "sindhi"])
    parser.add_argument("--vocab_size", type=int, default=16000)
    parser.add_argument("--train_stride", type=int, default=50,
                        help="Sample every Nth line of train.txt when counting tokens")
    args = parser.parse_args()

    data_dir = PROJECT_ROOT / args.language / "data" / "preprocessed"
    model = (PROJECT_ROOT / args.language / "tokenizer" /
             f"{args.language}_unigram_{args.vocab_size}.model")
    if not model.exists():
        logger.error(f"Tokenizer not found: {model}")
        return

    sp = spm.SentencePieceProcessor(model_file=str(model))

    stats = {"language": args.language, "tokenizer": model.name, "splits": {}}
    for split in SPLITS:
        path = data_dir / f"{split}.txt"
        if not path.exists():
            logger.warning(f"Missing split file: {path}")
            continue
        stride = args.train_stride if split == "train" else 1
        logger.info(f"Measuring {split}.txt (token stride {stride})...")
        stats["splits"][split] = measure_split(path, sp, token_stride=stride)

    totals = {k: sum(s[k] for s in stats["splits"].values())
              for k in ("documents", "content_lines", "bytes", "tokens")}
    stats["totals"] = totals

    # Realised split proportions, which drift from the nominal 95/2.5/2.5 because the
    # MinHash pass was run independently per split and pruned each by a different amount.
    stats["realised_token_split"] = {
        k: round(v["tokens"] / totals["tokens"], 4) for k, v in stats["splits"].items()
    }

    out = data_dir / "split_statistics.json"
    out.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    logger.info(f"Wrote {out}")

    for split, s in stats["splits"].items():
        logger.info(f"  {split:5s} {s['megabytes']:>9.1f} MB  {s['documents']:>9,} docs  "
                    f"{s['tokens']:>13,} tokens"
                    f"{'' if s['tokens_exact'] else ' (estimated)'}")
    logger.info(f"  TOTAL {totals['bytes']/1024/1024:>9.1f} MB  {totals['documents']:>9,} docs  "
                f"{totals['tokens']:>13,} tokens")


if __name__ == "__main__":
    main()
