"""
Attention analysis (Step 10): what did the heads actually learn?

Computation only — plotting lives in plot_attention.py, so the numbers can be recomputed
and checked without a display, and the figures can be restyled without re-running the
model.

Three measurements per (layer, head):

  entropy          how spread out a head's attention is
  mean distance    how far back it typically looks
  positional focus how much mass sits on self and on the immediately previous token

Together these separate *positional* heads, which attend by offset regardless of content,
from *content-based* heads, which vary with what the tokens actually are.

Usage:
    python -m common.eval.attention --language hindi
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
import sentencepiece as spm

from common.model.config import GPTConfig
from common.model.gpt import GPT

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("attention")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DOC_SEPARATOR = "<|endoftext|>"

# Query positions before this index are skipped in the aggregate statistics. A causal mask
# means query t can only see t+1 keys, so early queries are structurally forced to be
# low-entropy and short-range. Including them would confound "this head is local" with
# "this head had nowhere else to look", and would bias the two models differently because
# their context lengths differ.
MIN_QUERY_POS = 32


def attention_statistics(attn: torch.Tensor, min_query: int = MIN_QUERY_POS) -> dict:
    """
    Reduce one layer's attention probabilities to per-head statistics.

    Args:
        attn: (B, n_heads, T, T) post-softmax probabilities. Row t sums to 1 over keys
            0..t; entries above the diagonal are exactly zero because of the causal mask.
        min_query: ignore query positions below this index.

    Returns per-head arrays, each of length n_heads:
        entropy_bits       mean Shannon entropy of the attention distribution
        entropy_normalised entropy divided by log2(t+1), the maximum available at that
                           position. 1.0 means uniform over everything visible, 0.0 means
                           all mass on a single key. This is the comparable number across
                           positions and across the two models' different context lengths.
        mean_distance      mean of sum_j p(t,j) * (t - j), in tokens
        self_attention     mean mass on the query's own position
        prev_token         mean mass on position t-1
    """
    if attn.dim() != 4:
        raise ValueError(f"expected (B, H, T, T), got {tuple(attn.shape)}")
    B, H, T, T2 = attn.shape
    if T != T2:
        raise ValueError(f"attention must be square in T, got {T}x{T2}")
    if T <= min_query:
        raise ValueError(f"sequence length {T} leaves no queries above min_query={min_query}")

    a = attn.float()
    positions = torch.arange(T, device=a.device)

    # Entropy per (batch, head, query). Zeros contribute nothing: 0*log0 -> 0.
    safe = a.clamp_min(1e-12)
    ent = -(a * safe.log2()).sum(-1)                       # (B, H, T)

    # Maximum entropy available at query t is log2(t+1) visible keys. Position 0 has a
    # single option, so its maximum is 0 and the ratio is undefined; it is excluded by
    # min_query anyway.
    max_ent = torch.log2((positions + 1).float()).clamp_min(1e-12)  # (T,)
    ent_norm = ent / max_ent

    # Distance from each query to each key, only meaningful on the causal (lower) triangle.
    dist = (positions.view(-1, 1) - positions.view(1, -1)).float()  # (T, T)
    mean_dist = (a * dist).sum(-1)                                  # (B, H, T)

    self_mass = a.diagonal(dim1=-2, dim2=-1)                        # (B, H, T)
    # a.diagonal(offset=-1) is exactly the entries a[..., t, t-1] for t = 1..T-1, i.e.
    # length T-1. Slicing the tensor first AND taking an offset diagonal would shorten it
    # twice and misalign every value by one position.
    prev_mass = torch.zeros_like(self_mass)
    prev_mass[..., 1:] = a.diagonal(offset=-1, dim1=-2, dim2=-1)

    sl = slice(min_query, None)
    return {
        "entropy_bits": ent[..., sl].mean(dim=(0, 2)).cpu().numpy(),
        "entropy_normalised": ent_norm[..., sl].mean(dim=(0, 2)).cpu().numpy(),
        "mean_distance": mean_dist[..., sl].mean(dim=(0, 2)).cpu().numpy(),
        "self_attention": self_mass[..., sl].mean(dim=(0, 2)).cpu().numpy(),
        "prev_token": prev_mass[..., sl].mean(dim=(0, 2)).cpu().numpy(),
    }


def sample_clean_windows(tokens: np.ndarray, sep_id: int, n: int, length: int,
                         seed: int = 0) -> np.ndarray:
    """
    Draw windows that contain no document separator.

    A separator is a hard topic boundary. Heads attend to it strongly and idiosyncratically,
    which would distort every statistic here — especially for Sindhi, whose splits carry
    far more separators than Hindi's.
    """
    rng = np.random.default_rng(seed)
    out, attempts = [], 0
    while len(out) < n and attempts < n * 200:
        attempts += 1
        s = int(rng.integers(0, len(tokens) - length))
        w = np.asarray(tokens[s:s + length])
        if (w == sep_id).any():
            continue
        out.append(w)
    if len(out) < n:
        logger.warning("only %d/%d separator-free windows found", len(out), n)
    return np.stack(out)


@torch.no_grad()
def analyse(language: str, split: str, n_windows: int, seq_len: int,
            device: str, seed: int, checkpoint: str = "best") -> dict:
    """Run the model over held-out text and collect per-layer, per-head statistics."""
    from common.eval.intrinsic import checkpoint_path
    ckpt_path = checkpoint_path(language, checkpoint)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = GPTConfig(**ckpt["model_config"])
    model = GPT(cfg)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    sp = spm.SentencePieceProcessor(
        model_file=str(PROJECT_ROOT / language / "tokenizer"
                       / f"{language}_unigram_16000.model"))
    sep_id = sp.piece_to_id(DOC_SEPARATOR)

    data = np.memmap(PROJECT_ROOT / language / "data" / "tokenized" / f"{split}.bin",
                     dtype=np.uint16, mode="r")
    seq_len = min(seq_len, cfg.max_seq_len)
    windows = sample_clean_windows(data, sep_id, n_windows, seq_len, seed)
    logger.info("%s: %d windows of %d tokens, %d layers x %d heads",
                language, len(windows), seq_len, cfg.n_layers, cfg.n_heads)

    idx = torch.tensor(windows.astype(np.int64), device=device)
    # Attention maps are (B, H, T, T) per layer and dominate memory, so the batch is fed
    # in small chunks and statistics are accumulated rather than stored.
    per_layer = [[] for _ in range(cfg.n_layers)]
    chunk = max(1, 2048 // seq_len)
    for i in range(0, len(idx), chunk):
        _, _, attns = model(idx[i:i + chunk], return_attention=True)
        for layer, a in enumerate(attns):
            per_layer[layer].append(attention_statistics(a))

    stats = {}
    for layer, chunks in enumerate(per_layer):
        stats[layer] = {k: np.mean([c[k] for c in chunks], axis=0).tolist()
                        for k in chunks[0]}

    # Keep one example window's raw maps for the heatmaps, with its decoded text so the
    # figure can be labelled with real tokens in the model's own script.
    example_ids = windows[0][:min(48, seq_len)]
    _, _, ex_attns = model(
        torch.tensor(example_ids.astype(np.int64), device=device).unsqueeze(0),
        return_attention=True)

    return {
        "language": language,
        "split": split,
        "n_layers": cfg.n_layers,
        "n_heads": cfg.n_heads,
        "context": cfg.max_seq_len,
        "seq_len": seq_len,
        "n_windows": len(windows),
        "min_query_pos": MIN_QUERY_POS,
        "layers": stats,
        "example": {
            "token_ids": example_ids.tolist(),
            "tokens": [sp.id_to_piece(int(t)) for t in example_ids],
            "text": sp.decode(example_ids.tolist()),
            "attention": [a[0].float().cpu().numpy().tolist() for a in ex_attns],
        },
    }


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--language", choices=["hindi", "sindhi"], help="default: both")
    p.add_argument("--split", default="val", choices=["val", "test"])
    p.add_argument("--n_windows", type=int, default=64)
    p.add_argument("--seq_len", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    # Defaults reproduce Phase 2 exactly. The bonus ablation passes --checkpoint nopos and
    # its own --out so the Phase 2 results file is never overwritten.
    p.add_argument("--checkpoint", default="best",
                   help="best | finetuned | nopos | path — see intrinsic.checkpoint_path")
    p.add_argument("--out", default=None, help="default: report/attention_results.json")
    args = p.parse_args()

    out = [analyse(lang, args.split, args.n_windows, args.seq_len, args.device, args.seed,
                   args.checkpoint)
           for lang in ([args.language] if args.language else ["hindi", "sindhi"])]

    path = Path(args.out) if args.out else PROJECT_ROOT / "report" / "attention_results.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    logger.info("Wrote %s", path)

    for r in out:
        logger.info("%s: entropy (normalised) and mean distance by layer", r["language"])
        for layer in range(r["n_layers"]):
            s = r["layers"][layer]
            logger.info("  L%d  ent %s", layer,
                        " ".join(f"{v:.2f}" for v in s["entropy_normalised"]))
            logger.info("      dist %s", " ".join(f"{v:5.1f}" for v in s["mean_distance"]))


if __name__ == "__main__":
    main()
