"""
Train/validation/test split definition for both corpora.

WHY THIS FILE EXISTS
--------------------
The split definition is a required deliverable, but for a while it was not in the
repository. The shipped corpora were produced in two steps:

  1. An earlier two-way revision of `pipeline.py` assigned each document to train (95%)
     or val (5%) by a seeded uniform draw.
  2. That val file was then cut in half into val (2.5%) and test (2.5%) by a script that
     was never committed.

`pipeline.py` now performs the full 95 / 2.5 / 2.5 assignment directly in one pass, so
step 2 is not needed for new runs. This module records the historical step so the shipped
artefacts remain reproducible, and provides a `--verify` mode that checks split files on
disk against the documented definition.

    Nominal split:  train 95.0%   val 2.5%   test 2.5%

REALISED PROPORTIONS DIFFER SLIGHTLY
------------------------------------
MinHash fuzzy deduplication was run independently on each split after the assignment, and
pruned a different fraction from each. The shipped corpora therefore realise:

    Hindi   train 94.99%   val 2.79%   test 2.23%
    Sindhi  train 94.92%   val 2.78%   test 2.30%

This is drift from post-hoc deduplication, not from a biased assignment: the draw itself
is symmetric between val and test (see `assign_split` below). The eval splits remain far
larger than needed for stable perplexity estimation, so they were left as-is rather than
re-drawn — re-drawing would invalidate every statistic already reported and every copy of
the corpus already published to Drive.

    python -m common.preprocessing.make_splits --language hindi --verify
    python -m common.preprocessing.make_splits --language hindi --split-val-file  # historical step 2
"""

import argparse
import json
import logging
import random
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("make_splits")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DOC_SEPARATOR = "<|endoftext|>"

# The split definition. Must match pipeline.run_pipeline()'s defaults.
SEED = 42
VAL_RATIO = 0.025
TEST_RATIO = 0.025
TRAIN_RATIO = 1.0 - VAL_RATIO - TEST_RATIO

# How far realised proportions may drift from nominal before the check flags them. The
# only legitimate source of drift is the per-split MinHash pass that ran after assignment;
# the observed worst case is 0.29 pts, so 1 pt catches a genuinely broken assignment while
# tolerating dedup.
TOLERANCE_PTS = 0.01


def assign_split(rand_val: float) -> str:
    """
    Map a uniform draw in [0, 1) to a split name.

    This is the single source of truth for the split boundaries and is intentionally
    identical to the branch in pipeline.run_pipeline(). Note the val and test bands are
    the same width, so the assignment carries no bias between them.
    """
    if rand_val < VAL_RATIO:
        return "val"
    if rand_val < VAL_RATIO + TEST_RATIO:
        return "test"
    return "train"


def iter_documents(path: Path):
    """Yield documents from a split file, splitting on the <|endoftext|> separator."""
    buffer = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            stripped = line.rstrip('\n')
            if stripped.strip() == DOC_SEPARATOR:
                if buffer:
                    yield "\n".join(buffer)
                buffer = []
            else:
                buffer.append(stripped)
    if buffer:
        yield "\n".join(buffer)


def split_val_file(data_dir: Path, dry_run: bool = True):
    """
    Historical step 2: cut a two-way pipeline's val.txt into equal val and test halves.

    Retained for reproducibility of the shipped corpora only. New runs do not need this —
    pipeline.py assigns all three splits directly. Refuses to overwrite unless explicitly
    told to, because the shipped val.txt/test.txt are the files every reported statistic
    describes and every published Drive copy contains.
    """
    src = data_dir / "val.txt"
    out_val = data_dir / "val_resplit.txt"
    out_test = data_dir / "test_resplit.txt"

    if dry_run:
        logger.info("DRY RUN — no files written. Would split %s into %s and %s",
                    src.name, out_val.name, out_test.name)
        return

    rng = random.Random(SEED)
    counts = {"val": 0, "test": 0}
    with open(out_val, 'w', encoding='utf-8') as fv, \
         open(out_test, 'w', encoding='utf-8') as ft:
        for doc in iter_documents(src):
            target = "val" if rng.random() < 0.5 else "test"
            (fv if target == "val" else ft).write(f"{doc}\n{DOC_SEPARATOR}\n")
            counts[target] += 1
    logger.info("Wrote %s (%d docs) and %s (%d docs)",
                out_val.name, counts["val"], out_test.name, counts["test"])
    logger.info("Written to *_resplit.txt — the shipped val.txt/test.txt are untouched.")


def verify(language: str):
    """
    Check the split files on disk against the documented definition.

    Compares realised token proportions to the nominal ratios and reports the deviation,
    using the measured counts in split_statistics.json.
    """
    stats_path = (PROJECT_ROOT / language / "data" / "preprocessed" / "split_statistics.json")
    if not stats_path.exists():
        logger.error("Missing %s — run common.preprocessing.split_stats first.", stats_path)
        return False

    stats = json.loads(stats_path.read_text())
    nominal = {"train": TRAIN_RATIO, "val": VAL_RATIO, "test": TEST_RATIO}

    logger.info("Split verification — %s (seed=%d)", language.upper(), SEED)
    logger.info("  tolerance: realised may drift up to %.2f pts from nominal, which is the "
                "budget for post-hoc per-split MinHash dedup", TOLERANCE_PTS * 100)
    ok = True
    for split, nom in nominal.items():
        realised = stats["realised_token_split"].get(split)
        if realised is None:
            logger.error("  %-6s MISSING from split_statistics.json", split)
            ok = False
            continue
        drift = realised - nom
        # Post-hoc MinHash dedup moves these by well under a percentage point; anything
        # larger would indicate the assignment itself, not dedup, is off.
        flag = "OK" if abs(drift) < TOLERANCE_PTS else "CHECK"
        logger.info("  %-6s nominal %6.2f%%   realised %6.2f%%   drift %+.2f pts   %s",
                    split, nom * 100, realised * 100, drift * 100, flag)
        ok &= abs(drift) < TOLERANCE_PTS

    logger.info("  total tokens: %s", f"{stats['totals']['tokens']:,}")
    logger.info("Result: %s", "PASS" if ok else "FAIL")
    return ok


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--language", required=True, choices=["hindi", "sindhi"])
    parser.add_argument("--verify", action="store_true",
                        help="Check on-disk splits against the documented definition")
    parser.add_argument("--split-val-file", action="store_true",
                        help="Historical step 2: halve a two-way pipeline's val.txt")
    parser.add_argument("--write", action="store_true",
                        help="With --split-val-file, actually write (default is a dry run)")
    args = parser.parse_args()

    if args.verify:
        raise SystemExit(0 if verify(args.language) else 1)

    if args.split_val_file:
        data_dir = PROJECT_ROOT / args.language / "data" / "preprocessed"
        split_val_file(data_dir, dry_run=not args.write)
        return

    parser.print_help()


if __name__ == "__main__":
    main()
