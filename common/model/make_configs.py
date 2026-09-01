"""
Emit the per-language model configuration files.

Model H (Hindi) and Model L (Sindhi) share an architecture in every respect except
context length. Generating both from one place means the two configs cannot silently
drift apart on the dimensions that are supposed to be identical.

Usage:
    python -m common.model.make_configs            # write both
    python -m common.model.make_configs --check    # verify on-disk configs match
"""

import argparse
import json
import logging
from pathlib import Path

import sentencepiece as spm

from common.model.config import GPTConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("make_configs")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Everything the two models hold in common. Only max_seq_len differs, and only for the
# documented corpus-shape reason below.
SHARED = dict(
    d_model=512,
    n_layers=6,
    n_heads=8,
    d_ff=1536,
    dropout=0.1,
    tie_embeddings=True,
)

# Context length per language. Measured document lengths on the held-out split:
#
#     Hindi   median   64 tokens, mean  85 -> 99.3% of documents fit in 512
#     Sindhi  median  219 tokens, mean 774 -> 74.5% of documents fit in 512,
#                                             86.4% fit in 1024
#
# Hindi documents are short news and encyclopaedia paragraphs, so a 512-token window
# already spans roughly eight of them; the extra context of a longer window would be
# filled with unrelated neighbouring articles rather than usable history. Sindhi documents
# are long-form books and articles at 3.4x the median length, so a 512-token window
# frequently sits inside a single continuous document with the earlier half of the same
# passage cut off. Longer context there is genuine, same-topic history.
CONTEXT = {"hindi": 512, "sindhi": 1024}

LANGUAGES = ("hindi", "sindhi")


def build(language: str) -> GPTConfig:
    """Construct the config for one language, reading vocab size from its tokenizer."""
    tokenizer_path = (PROJECT_ROOT / language / "tokenizer"
                      / f"{language}_unigram_16000.model")
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"tokenizer not found: {tokenizer_path}")

    sp = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
    return GPTConfig(
        vocab_size=sp.get_piece_size(),
        max_seq_len=CONTEXT[language],
        language=language,
        tokenizer=tokenizer_path.name,
        **SHARED,
    )


def config_path(language: str) -> Path:
    return PROJECT_ROOT / language / "configs" / "model_config.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true",
                        help="Verify the on-disk configs match what this module would emit")
    args = parser.parse_args()

    ok = True
    for language in LANGUAGES:
        cfg = build(language)
        path = config_path(language)

        if args.check:
            if not path.exists():
                logger.error("  %s: missing", path)
                ok = False
                continue
            on_disk = GPTConfig.from_json(path)
            if on_disk != cfg:
                logger.error("  %s: on-disk config differs from generated", path)
                ok = False
            else:
                logger.info("  %s: matches", path)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            cfg.to_json(path)
            logger.info("  wrote %s", path)

        logger.info(
            "    %-6s vocab=%s ctx=%-5s params=%s (non-embedding %s) d_head=%s",
            cfg.language, f"{cfg.vocab_size:,}", cfg.max_seq_len,
            f"{cfg.n_params():,}", f"{cfg.n_params(non_embedding=True):,}", cfg.d_head)

    # The architecture must be identical apart from context length; assert it rather than
    # trusting the SHARED dict to have been used consistently.
    a, b = build("hindi"), build("sindhi")
    for field in SHARED:
        if getattr(a, field) != getattr(b, field):
            logger.error("  %s differs between models: %s vs %s",
                         field, getattr(a, field), getattr(b, field))
            ok = False
    delta = b.n_params() - a.n_params()
    expected = (b.max_seq_len - a.max_seq_len) * a.d_model
    logger.info("  parameter difference: %s (entirely the position table: %s)",
                f"{delta:,}", f"{expected:,}")
    if delta != expected:
        logger.error("  models differ by more than the position table — check SHARED")
        ok = False

    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
