"""
Intrinsic language-model evaluation: cross-entropy, perplexity, and bits-per-byte.

    python -m common.eval.intrinsic --language hindi
    python -m common.eval.intrinsic --all            # both models, native + matched context

Why three numbers rather than one
---------------------------------
* Cross-entropy is what the model was trained to minimise, in nats per token.
* Perplexity is exp(cross-entropy). It has an intuitive reading: the model is as uncertain
  as if it were choosing uniformly among this many tokens at each position.
* Bits-per-byte normalises by the UTF-8 bytes of the underlying text instead of by tokens.

The third exists because the first two are NOT comparable across our two models. Hindi and
Sindhi have separate 16,000-entry vocabularies learned independently, so a "token" means a
different amount of text in each. A model whose tokenizer packs more characters into each
token faces fewer, harder predictions and can show a worse perplexity while actually
compressing the language better. BPB removes that: bits needed per byte of original text
is a property of the text and the model, not of the tokenizer's granularity. It is the
only figure in this file on which Model H and Model L may be compared directly.
"""

import argparse
import json
import logging
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import torch
import sentencepiece as spm

from common.data.loader import TokenDataset
from common.model.config import GPTConfig
from common.model.gpt import GPT
from common.train.trainer import select_precision

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("intrinsic")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# The context both models can be compared at: the shorter of the two, so neither is
# conditioned on more history than the other.
MATCHED_CONTEXT = 512


@dataclass
class EvalResult:
    """One (model, split, context) measurement."""
    language: str
    split: str
    context: int
    cross_entropy: float      # nats per token
    perplexity: float
    bits_per_byte: float
    n_tokens: int
    n_bytes: int
    n_windows: int

    def row(self) -> str:
        return (f"  {self.language:7s} {self.split:5s} ctx {self.context:>4}  "
                f"CE {self.cross_entropy:6.4f}  PPL {self.perplexity:8.2f}  "
                f"BPB {self.bits_per_byte:6.4f}  "
                f"({self.n_tokens:,} tokens / {self.n_bytes:,} bytes)")


@torch.no_grad()
def evaluate(
    model: GPT,
    dataset: TokenDataset,
    tokenizer: spm.SentencePieceProcessor,
    device: str,
    batch_size: int = 8,
    max_windows: Optional[int] = None,
) -> tuple[float, float, int, int, int]:
    """
    Score a split deterministically and return (total_nats, total_bytes, tokens, windows).

    Uses non-overlapping sequential windows rather than random sampling, so every token is
    scored exactly once and the result is a fixed property of the split rather than an
    estimate that moves between runs.

    Returns:
        (sum_nats, n_tokens, n_bytes, n_windows, unused) — see callers.
    """
    model.eval()
    amp_dtype, _ = select_precision(device)

    total_nats = 0.0
    total_tokens = 0
    total_bytes = 0
    n_windows = 0

    for x, y in dataset.iter_sequential(batch_size, device):
        if max_windows is not None and n_windows >= max_windows:
            break

        with torch.autocast("cuda", dtype=amp_dtype,
                            enabled=device.startswith("cuda")):
            logits, _ = model(x)

        # reduction="sum" in float32: summing hundreds of thousands of per-token losses in
        # bf16 would accumulate rounding error large enough to move the fourth decimal.
        loss_sum = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)).float(),
            y.reshape(-1),
            reduction="sum",
        )
        total_nats += loss_sum.item()
        total_tokens += y.numel()
        n_windows += x.size(0)

        # Bytes of the text the scored targets represent. Decoded per row rather than per
        # token, because SentencePiece pieces only reassemble into correct text (with word
        # boundaries) as a sequence.
        for row in y.tolist():
            total_bytes += len(tokenizer.decode(row).encode("utf-8"))

    return total_nats, total_tokens, total_bytes, n_windows


def evaluate_model(
    language: str,
    split: str,
    context: Optional[int] = None,
    checkpoint: str = "best",
    device: str = "cuda",
    batch_size: int = 8,
    max_windows: Optional[int] = None,
) -> EvalResult:
    """
    Load a trained model and measure it on one split.

    Args:
        context: sequence length to evaluate at. Defaults to the model's own max_seq_len.
            Passing a smaller value evaluates the model on shorter windows, which is how
            the matched-context comparison is done: the position embeddings are simply
            sliced, so the model runs unmodified.
    """
    run_dir = PROJECT_ROOT / language / "train" / "run"
    ckpt = torch.load(run_dir / f"{checkpoint}.pt", map_location="cpu", weights_only=False)
    cfg = GPTConfig(**ckpt["model_config"])

    if cfg.language != language:
        raise ValueError(f"checkpoint says language={cfg.language}, expected {language}")

    block = context or cfg.max_seq_len
    if block > cfg.max_seq_len:
        raise ValueError(f"cannot evaluate at context {block}: the model's position table "
                         f"only has {cfg.max_seq_len} rows")

    model = GPT(cfg)
    model.load_state_dict(ckpt["model"])
    model.to(device)

    tokenizer = spm.SentencePieceProcessor(
        model_file=str(PROJECT_ROOT / language / "tokenizer" / cfg.tokenizer))
    if tokenizer.get_piece_size() != cfg.vocab_size:
        raise ValueError("tokenizer vocabulary does not match the checkpoint")

    data = TokenDataset(
        PROJECT_ROOT / language / "data" / "tokenized" / f"{split}.bin", block_size=block)

    logger.info("evaluating %s %s at context %d (%s tokens available)",
                language, split, block, f"{data.n_tokens:,}")

    nats, n_tokens, n_bytes, n_windows = evaluate(
        model, data, tokenizer, device, batch_size, max_windows)

    ce = nats / n_tokens
    return EvalResult(
        language=language,
        split=split,
        context=block,
        cross_entropy=ce,
        perplexity=math.exp(ce),
        # nats -> bits, then per byte of the underlying UTF-8 text.
        bits_per_byte=(nats / math.log(2)) / n_bytes,
        n_tokens=n_tokens,
        n_bytes=n_bytes,
        n_windows=n_windows,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--language", choices=["hindi", "sindhi"])
    p.add_argument("--all", action="store_true",
                   help="both models, both splits, native and matched context")
    p.add_argument("--splits", nargs="+", default=["val", "test"])
    p.add_argument("--checkpoint", default="best")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--max_windows", type=int, default=None,
                   help="cap windows per measurement (for a quick smoke test)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default=None, help="write results JSON here")
    args = p.parse_args()

    if not args.all and not args.language:
        p.error("pass --language or --all")

    languages = ["hindi", "sindhi"] if args.all else [args.language]
    results: list[EvalResult] = []

    for language in languages:
        for split in args.splits:
            # Native context: how the model actually behaves.
            results.append(evaluate_model(
                language, split, None, args.checkpoint, args.device,
                args.batch_size, args.max_windows))
            # Matched context, only where it differs from native, so the two models are
            # conditioned on identical amounts of history.
            cfg = GPTConfig(**torch.load(
                PROJECT_ROOT / language / "train" / "run" / f"{args.checkpoint}.pt",
                map_location="cpu", weights_only=False)["model_config"])
            if cfg.max_seq_len != MATCHED_CONTEXT:
                results.append(evaluate_model(
                    language, split, MATCHED_CONTEXT, args.checkpoint, args.device,
                    args.batch_size, args.max_windows))

    print("\n" + "=" * 78)
    print("INTRINSIC EVALUATION")
    print("=" * 78)
    for r in results:
        print(r.row())

    # The headline comparison, at matched context and in BPB.
    matched = {r.language: r for r in results
               if r.split == "test" and r.context == MATCHED_CONTEXT}
    if len(matched) == 2:
        h, s = matched["hindi"], matched["sindhi"]
        print("\n" + "-" * 78)
        print(f"Matched-context test comparison (both at {MATCHED_CONTEXT} tokens):")
        print(f"  bits per byte : Hindi {h.bits_per_byte:.4f}  vs  "
              f"Sindhi {s.bits_per_byte:.4f}  "
              f"({(s.bits_per_byte/h.bits_per_byte - 1):+.1%} for Model L)")
        print(f"  perplexity    : Hindi {h.perplexity:.2f}  vs  Sindhi {s.perplexity:.2f}")
        print("  BPB is the comparable figure; perplexity is shown for reference only,")
        print("  since the two tokenizers segment their languages differently.")

    if args.out:
        Path(args.out).write_text(
            json.dumps([asdict(r) for r in results], indent=2) + "\n", encoding="utf-8")
        logger.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
