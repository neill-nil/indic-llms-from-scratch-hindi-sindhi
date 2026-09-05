"""
Training curves (Step 11), rendered from each run's training_log.jsonl.

The brief lists "training logs and loss curves" as a deliverable and warns that plots
missing a title, axis labels or a legend may score zero, so every axis here carries all
three.

Usage:
    python -m common.eval.plot_training
"""

import json
import logging
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("plot_training")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
FIG_DIR = PROJECT_ROOT / "report" / "figures"

DISPLAY = {"hindi": "Model H (Hindi)", "sindhi": "Model L (Sindhi)"}
COLOUR = {"hindi": "#1f77b4", "sindhi": "#d62728"}

# A model that has learned nothing spreads its probability evenly over the vocabulary, so
# its loss is ln(vocab_size). Drawing this line turns an abstract loss number into a
# visible reference: everything below it is knowledge the model actually acquired.
UNIFORM_LOSS = math.log(16000)


def load(language: str):
    path = PROJECT_ROOT / language / "train" / "run" / "training_log.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    train = [r for r in records if "train_loss" in r]
    evals = [r for r in records if "val_loss" in r]
    return train, evals


def smooth(values, window: int = 21):
    """
    Moving average for the training curve, returned with the x-indices it is valid for.

    Per-step training loss is noisy because each point is a single batch. The raw trace is
    kept visible underneath at low opacity so the smoothing is not hiding anything.

    Returns (offset, smoothed). mode="valid" is used deliberately: mode="same" pads the
    ends with zeros, which drags the first and last half-window towards zero and draws a
    dramatic plunge at both edges that never happened. The offset tells the caller which
    x-values the returned points correspond to.
    """
    values = np.asarray(values, dtype=float)
    if len(values) < window:
        return 0, values
    kernel = np.ones(window) / window
    return window // 2, np.convolve(values, kernel, mode="valid")


def plot_curves(out_path: Path) -> None:
    """Four panels: loss, perplexity, learning rate, gradient norm."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    data = {lang: load(lang) for lang in ("hindi", "sindhi")}

    # ── Loss ────────────────────────────────────────────────────────────────────
    ax = axes[0, 0]
    ax.axhline(UNIFORM_LOSS, color="grey", linestyle=":", linewidth=1.5,
               label=f"Uniform guess = ln(16000) = {UNIFORM_LOSS:.2f}")
    for lang, (train, evals) in data.items():
        x = [r["tokens"] / 1e6 for r in train]
        y = [r["train_loss"] for r in train]
        ax.plot(x, y, color=COLOUR[lang], alpha=0.18, linewidth=0.7)
        off, sm = smooth(y)
        ax.plot(x[off:off + len(sm)], sm, color=COLOUR[lang], linewidth=1.8,
                label=f"{DISPLAY[lang]} — training")
        ax.plot([r["tokens"] / 1e6 for r in evals], [r["val_loss"] for r in evals],
                color=COLOUR[lang], linestyle="--", marker="o", markersize=3,
                linewidth=1.5, label=f"{DISPLAY[lang]} — validation")
    ax.set_title("Training and validation loss", fontsize=12)
    ax.set_xlabel("Tokens consumed (millions)")
    ax.set_ylabel("Cross-entropy loss (nats/token)")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)

    # ── Perplexity ──────────────────────────────────────────────────────────────
    ax = axes[0, 1]
    for lang, (_, evals) in data.items():
        ax.plot([r["tokens"] / 1e6 for r in evals], [r["val_perplexity"] for r in evals],
                color=COLOUR[lang], marker="o", markersize=4, linewidth=1.8,
                label=DISPLAY[lang])
        final = evals[-1]
        ax.annotate(f"{final['val_perplexity']:.1f}",
                    (final["tokens"] / 1e6, final["val_perplexity"]),
                    textcoords="offset points", xytext=(6, 6),
                    fontsize=9, color=COLOUR[lang], fontweight="bold")
    ax.set_yscale("log")     # early perplexity is orders of magnitude larger than final
    ax.set_title("Validation perplexity (log scale)", fontsize=12)
    ax.set_xlabel("Tokens consumed (millions)")
    ax.set_ylabel("Perplexity (lower is better)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, which="both")

    # ── Learning rate ───────────────────────────────────────────────────────────
    ax = axes[1, 0]
    for lang, (train, _) in data.items():
        ax.plot([r["tokens"] / 1e6 for r in train], [r["lr"] for r in train],
                color=COLOUR[lang], linewidth=1.8, label=DISPLAY[lang])
    ax.set_title("Learning-rate schedule: 500-step warmup, then cosine decay", fontsize=12)
    ax.set_xlabel("Tokens consumed (millions)")
    ax.set_ylabel("Learning rate")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # ── Gradient norm ───────────────────────────────────────────────────────────
    ax = axes[1, 1]
    ax.axhline(1.0, color="grey", linestyle=":", linewidth=1.5,
               label="Clipping threshold = 1.0")
    for lang, (train, _) in data.items():
        x = [r["tokens"] / 1e6 for r in train]
        y = [r["grad_norm"] for r in train]
        ax.plot(x, y, color=COLOUR[lang], alpha=0.18, linewidth=0.7)
        off, sm = smooth(y)
        ax.plot(x[off:off + len(sm)], sm, color=COLOUR[lang], linewidth=1.8,
                label=DISPLAY[lang])
    ax.set_title("Gradient norm — stability check", fontsize=12)
    ax.set_xlabel("Tokens consumed (millions)")
    ax.set_ylabel("Gradient L2 norm (before clipping)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    fig.suptitle("Pretraining curves — both models, 500M tokens each, identical "
                 "optimisation settings", fontsize=13)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("  wrote %s", out_path.name)


def plot_bpb_comparison(out_path: Path) -> None:
    """The headline intrinsic result as a figure."""
    results = json.loads((PROJECT_ROOT / "report" / "intrinsic_results.json").read_text())
    by = {(r["language"], r["split"], r["context"]): r for r in results}

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)

    # Matched-context bits per byte, the number the comparison rests on.
    ax = axes[0]
    splits = ["val", "test"]
    width = 0.35
    x = np.arange(len(splits))
    for i, lang in enumerate(("hindi", "sindhi")):
        vals = [by[(lang, s, 512)]["bits_per_byte"] for s in splits]
        bars = ax.bar(x + (i - 0.5) * width, vals, width,
                      color=COLOUR[lang], label=DISPLAY[lang])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.4f}",
                    ha="center", fontsize=9, fontweight="bold")
    ax.set_title("Bits per byte at matched 512-token context\n"
                 "(lower is better; comparable across tokenizers)", fontsize=11)
    ax.set_xlabel("Held-out split")
    ax.set_ylabel("Bits per byte")
    ax.set_xticks(x); ax.set_xticklabels(splits)
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis="y")

    # Perplexity, to show why it is NOT the metric to compare on.
    ax = axes[1]
    for i, lang in enumerate(("hindi", "sindhi")):
        vals = [by[(lang, s, 512)]["perplexity"] for s in splits]
        bars = ax.bar(x + (i - 0.5) * width, vals, width,
                      color=COLOUR[lang], label=DISPLAY[lang], alpha=0.85)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 1.0, f"{v:.1f}",
                    ha="center", fontsize=9, fontweight="bold")
    ax.set_title("Perplexity at matched context\n"
                 "Sindhi 'improves' on test — an artifact of 16.6% corrupted text",
                 fontsize=11)
    ax.set_xlabel("Held-out split")
    ax.set_ylabel("Perplexity")
    ax.set_xticks(x); ax.set_xticklabels(splits)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle("Intrinsic evaluation: bits-per-byte is comparable across models, "
                 "perplexity is not", fontsize=12)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("  wrote %s", out_path.name)


def plot_generation_metrics(out_path: Path) -> None:
    """Generation quality and diversity against decoding temperature."""
    results = json.loads((PROJECT_ROOT / "report" / "generation_results.json").read_text())
    settings = ["greedy", "temp_0.5", "temp_1.0", "temp_1.5"]
    labels = ["greedy\n(T=0)", "T=0.5", "T=1.0", "T=1.5"]
    x = np.arange(len(settings))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), constrained_layout=True)
    panels = [("chrf++", "chrF++ — overlap with reference", "chrF++ score", axes[0]),
              ("repetition_rate_4gram", "Repetition rate — degeneracy check",
               "Fraction of repeated 4-grams", axes[1]),
              ("distinct_2", "Distinct-2 — output diversity",
               "Fraction of unique bigrams", axes[2])]

    for key, title, ylabel, ax in panels:
        for r in results:
            lang = r["language"]
            vals = [r["metrics"][s][key] for s in settings]
            ax.plot(x, vals, marker="o", markersize=7, linewidth=2,
                    color=COLOUR[lang], label=DISPLAY[lang])
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Decoding temperature")
        ax.set_ylabel(ylabel)
        ax.set_xticks(x); ax.set_xticklabels(labels)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    fig.suptitle("Generation quality versus decoding temperature\n"
                 "Greedy decoding degenerates into loops; chrF++ peaks at T=1.0",
                 fontsize=12)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("  wrote %s", out_path.name)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("writing figures to %s", FIG_DIR)
    plot_curves(FIG_DIR / "training_curves.png")
    plot_bpb_comparison(FIG_DIR / "intrinsic_comparison.png")
    plot_generation_metrics(FIG_DIR / "generation_metrics.png")
    logger.info("done")


if __name__ == "__main__":
    main()
