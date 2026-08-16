"""
Tokenizer Training Script for Hindi and Sindhi Corpora.

Trains a SentencePiece tokenizer from scratch on preprocessed text,
supporting both BPE and Unigram model types.

Memory-efficient: uses Unix stream processing (grep + head) to prepare
the training text without loading everything into Python memory.

Usage:
    python -m common.tokenizer.train_tokenizer \
        --language hindi --model_type unigram --vocab_size 32000
    python -m common.tokenizer.train_tokenizer \
        --language sindhi --model_type bpe --vocab_size 32000
"""

import argparse
import logging
import os
import subprocess
import time
from pathlib import Path

import sentencepiece as spm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("tokenizer_training")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DOC_SEPARATOR = "<|endoftext|>"


def prepare_training_text(train_path: Path, out_path: Path, max_lines: int = 0):
    """
    Use Unix pipeline (grep + head) to strip document separators and blank
    lines from train.txt.  This avoids loading the file into Python memory
    and is extremely fast even on multi-GB files.
    """
    logger.info(f"Preparing training text from {train_path} ...")

    # grep -v: remove lines matching separator or empty
    # Use shell pipeline for speed
    if max_lines > 0:
        cmd = (
            f"grep -v -E '^\\s*$|^{DOC_SEPARATOR}$' '{train_path}' "
            f"| head -n {max_lines} > '{out_path}'"
        )
    else:
        cmd = (
            f"grep -v -E '^\\s*$|^{DOC_SEPARATOR}$' '{train_path}' "
            f"> '{out_path}'"
        )

    logger.info(f"  Running: {cmd[:120]}...")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode not in (0, 1):  # grep returns 1 if no match (fine)
        logger.error(f"  grep failed: {result.stderr}")
        return 0

    # Count lines written
    wc = subprocess.run(
        ["wc", "-l", str(out_path)], capture_output=True, text=True
    )
    line_count = int(wc.stdout.strip().split()[0]) if wc.returncode == 0 else 0
    logger.info(f"  Wrote {line_count:,} lines to {out_path.name}")
    return line_count


def train_sentencepiece(
    input_path: Path,
    output_dir: Path,
    language: str,
    model_type: str,
    vocab_size: int,
    character_coverage: float = 0.9995,
):
    """
    Train a SentencePiece model from scratch.

    Args:
        input_path: Path to cleaned training text (one sentence per line)
        output_dir: Directory to save model and vocab files
        language: 'hindi' or 'sindhi'
        model_type: 'unigram' or 'bpe'
        vocab_size: Target vocabulary size
        character_coverage: Fraction of characters to cover (0.9995 for non-Latin scripts)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    model_prefix = output_dir / f"{language}_{model_type}_{vocab_size}"

    logger.info(f"Training SentencePiece {model_type.upper()} tokenizer...")
    logger.info(f"  Language:           {language}")
    logger.info(f"  Model type:         {model_type}")
    logger.info(f"  Vocab size:         {vocab_size:,}")
    logger.info(f"  Char coverage:      {character_coverage}")
    logger.info(f"  Input:              {input_path}")
    logger.info(f"  Output prefix:      {model_prefix}")

    start = time.time()

    spm.SentencePieceTrainer.train(
        input=str(input_path),
        model_prefix=str(model_prefix),
        model_type=model_type,
        vocab_size=vocab_size,
        character_coverage=character_coverage,
        # Normalization
        normalization_rule_name='nfkc',
        # Special tokens — index 0=<pad>, 1=<unk>, 2=<s>, 3=</s>
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        # Add our document separator as a user-defined special token
        user_defined_symbols=[DOC_SEPARATOR],
        # Training parameters — sample 2M sentences to keep RAM < 2GB
        input_sentence_size=500_000,
        shuffle_input_sentence=True,
        # Byte fallback for rare characters (prevents <unk>)
        byte_fallback=True,
        # Split digits into individual tokens
        split_digits=True,
        # Number of threads (leave 2 cores free for OS/Antigravity)
        num_threads=max((os.cpu_count() or 4) - 2, 2),
    )

    elapsed = time.time() - start
    logger.info(f"Training complete in {elapsed:.1f}s")
    logger.info(f"  Model file: {model_prefix}.model")
    logger.info(f"  Vocab file: {model_prefix}.vocab")

    return str(model_prefix)


def main():
    parser = argparse.ArgumentParser(description="Train SentencePiece tokenizer from scratch")
    parser.add_argument("--language", required=True, choices=["hindi", "sindhi"],
                        help="Target language")
    parser.add_argument("--model_type", required=True, choices=["unigram", "bpe"],
                        help="SentencePiece model type")
    parser.add_argument("--vocab_size", type=int, default=32000,
                        help="Target vocabulary size (default: 32000)")
    parser.add_argument("--character_coverage", type=float, default=0.9995,
                        help="Character coverage (default: 0.9995)")
    parser.add_argument("--max_lines", type=int, default=0,
                        help="Max training lines (0 = use all, useful for quick tests)")
    args = parser.parse_args()

    train_path = PROJECT_ROOT / args.language / "data" / "preprocessed" / "train.txt"
    out_dir = PROJECT_ROOT / args.language / "tokenizer"

    if not train_path.exists():
        logger.error(f"Training data not found: {train_path}")
        return

    # Prepare clean training text (strip <|endoftext|> separators)
    clean_text_path = out_dir / "train_clean.txt"
    out_dir.mkdir(parents=True, exist_ok=True)
    prepare_training_text(train_path, clean_text_path, max_lines=args.max_lines)

    # Train the tokenizer
    model_prefix = train_sentencepiece(
        input_path=clean_text_path,
        output_dir=out_dir,
        language=args.language,
        model_type=args.model_type,
        vocab_size=args.vocab_size,
        character_coverage=args.character_coverage,
    )

    logger.info(f"\nTokenizer saved to: {model_prefix}.model")
    logger.info("Run evaluate_tokenizer.py to compare models.")


if __name__ == "__main__":
    main()
