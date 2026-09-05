"""
Attention figures (Step 10), rendered from attention_results.json.

Kept separate from the computation in attention.py: the numbers can be recomputed without
a display, and the figures can be restyled without re-running the model over held-out text.

Every figure carries a title, axis labels and either a legend or a labelled colourbar,
because the brief warns that plots missing them may score zero.

Usage:
    python -m common.eval.plot_attention
"""

import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # headless: no display on this machine
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("plot_attention")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS = PROJECT_ROOT / "report" / "attention_results.json"
FIG_DIR = PROJECT_ROOT / "report" / "figures"

DISPLAY = {"hindi": "Model H (Hindi)", "sindhi": "Model L (Sindhi)"}


def pick_font(language: str):
    """
    Find a font that can actually render the script, so tick labels are glyphs and not
    empty boxes. Devanagari and Perso-Arabic are absent from matplotlib's default DejaVu.
    """
    wanted = (["Noto Sans Devanagari", "Nirmala UI"] if language == "hindi"
              else ["Noto Naskh Arabic", "Noto Sans Arabic"])
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in wanted:
        if name in available:
            return name
    logger.warning("no %s font found; token labels may render as boxes", language)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1: attention heatmaps, early vs late layer
# ─────────────────────────────────────────────────────────────────────────────

def plot_heatmaps(result: dict, layer: int, out_path: Path, max_heads: int = 4,
                  n_tokens: int = 32) -> None:
    """
    Query-vs-key attention for several heads of one layer, on a real example sentence.

    The lower-triangular shape is the causal mask made visible: every cell above the
    diagonal is exactly zero because a position cannot attend to its own future.
    """
    lang = result["language"]
    attn = np.array(result["example"]["attention"][layer])      # (heads, T, T)
    tokens = result["example"]["tokens"][:n_tokens]
    attn = attn[:max_heads, :n_tokens, :n_tokens]
    font = pick_font(lang)

    n = attn.shape[0]
    fig, axes = plt.subplots(1, n, figsize=(4.1 * n, 4.6), constrained_layout=True)
    axes = np.atleast_1d(axes)

    for h, ax in enumerate(axes):
        # Scale to the 99th percentile of everything EXCEPT the first key column. Almost
        # all mass lands on position 0 (the "attention sink"), so scaling to the true
        # maximum saturates that one column and renders the rest of the matrix uniformly
        # dark, hiding the structure the figure exists to show. The sink still reads as
        # saturated bright; the remaining detail becomes visible.
        body = attn[h][:, 1:]
        vmax = float(np.percentile(body[body > 0], 99)) if (body > 0).any() else 1.0
        im = ax.imshow(attn[h], cmap="viridis", aspect="equal",
                       vmin=0.0, vmax=max(vmax, 1e-6), interpolation="nearest")
        ax.set_title(f"Head {h}", fontsize=11)
        ax.set_xlabel("Key position (attended to)", fontsize=9)
        if h == 0:
            ax.set_ylabel("Query position (attending from)", fontsize=9)
        step = max(1, n_tokens // 12)
        ticks = range(0, n_tokens, step)
        labels = [tokens[i].replace("▁", " ").strip() or "_" for i in ticks]
        ax.set_xticks(list(ticks))
        ax.set_xticklabels(labels, rotation=90, fontsize=6,
                           fontname=font if font else None)
        ax.set_yticks(list(ticks))
        ax.set_yticklabels(labels, fontsize=6, fontname=font if font else None)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Attention weight (clipped at p99)", fontsize=8)

    position = "early" if layer == 0 else "late"
    fig.suptitle(
        f"{DISPLAY[lang]} — attention heatmaps, layer {layer} ({position})\n"
        f"Lower-triangular structure is the causal mask: no position attends to its future.\n"
        f"Colour clipped at the 99th percentile so detail is visible past the position-0 sink",
        fontsize=12)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("  wrote %s", out_path.name)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2: entropy and distance per layer/head
# ─────────────────────────────────────────────────────────────────────────────

def plot_head_statistics(results: list, out_path: Path) -> None:
    """Per-head entropy and mean attention distance, both models side by side."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)

    for col, result in enumerate(results):
        lang = result["language"]
        L, H = result["n_layers"], result["n_heads"]
        ent = np.array([result["layers"][str(l)]["entropy_normalised"] for l in range(L)])
        dist = np.array([result["layers"][str(l)]["mean_distance"] for l in range(L)])

        ax = axes[0, col]
        im = ax.imshow(ent, cmap="magma", aspect="auto", vmin=0, vmax=1)
        ax.set_title(f"{DISPLAY[lang]} — normalised attention entropy", fontsize=11)
        ax.set_xlabel("Head index"); ax.set_ylabel("Layer index")
        ax.set_xticks(range(H)); ax.set_yticks(range(L))
        for l in range(L):
            for h in range(H):
                ax.text(h, l, f"{ent[l, h]:.2f}", ha="center", va="center",
                        fontsize=7, color="white" if ent[l, h] < 0.6 else "black")
        cb = fig.colorbar(im, ax=ax); cb.set_label("Entropy / max possible (1 = uniform)")

        ax = axes[1, col]
        im = ax.imshow(dist, cmap="viridis", aspect="auto")
        ax.set_title(f"{DISPLAY[lang]} — mean attention distance", fontsize=11)
        ax.set_xlabel("Head index"); ax.set_ylabel("Layer index")
        ax.set_xticks(range(H)); ax.set_yticks(range(L))
        for l in range(L):
            for h in range(H):
                ax.text(h, l, f"{dist[l, h]:.0f}", ha="center", va="center",
                        fontsize=7, color="white" if dist[l, h] < dist.max() * 0.6 else "black")
        cb = fig.colorbar(im, ax=ax); cb.set_label("Mean distance back (tokens)")

    fig.suptitle("Per-head attention behaviour: how diffuse, and how far back\n"
                 "Low entropy + short distance = positional head; "
                 "high entropy + long distance = content-based head", fontsize=12)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("  wrote %s", out_path.name)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3: how behaviour changes with depth
# ─────────────────────────────────────────────────────────────────────────────

def plot_depth_trends(results: list, out_path: Path) -> None:
    """Layer-wise means, so the two models can be compared directly."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), constrained_layout=True)
    colours = {"hindi": "#1f77b4", "sindhi": "#d62728"}
    markers = {"hindi": "o", "sindhi": "s"}

    panels = [
        ("entropy_normalised", "Normalised entropy", "Entropy / max possible", axes[0]),
        ("mean_distance", "Mean attention distance", "Distance back (tokens)", axes[1]),
        ("prev_token", "Mass on the previous token", "P(attend to t-1)", axes[2]),
    ]
    for key, title, ylabel, ax in panels:
        for result in results:
            lang = result["language"]
            L = result["n_layers"]
            vals = [np.mean(result["layers"][str(l)][key]) for l in range(L)]
            ax.plot(range(L), vals, marker=markers[lang], color=colours[lang],
                    label=DISPLAY[lang], linewidth=2, markersize=7)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Layer index (0 = first)")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)

    fig.suptitle("Attention behaviour across depth, averaged over heads", fontsize=12)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("  wrote %s", out_path.name)


def main():
    if not RESULTS.exists():
        logger.error("Missing %s — run common.eval.attention first.", RESULTS)
        raise SystemExit(1)

    results = json.loads(RESULTS.read_text())
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("writing figures to %s", FIG_DIR)

    for result in results:
        lang = result["language"]
        last = result["n_layers"] - 1
        for layer in (0, last):              # brief requires an early and a late layer
            plot_heatmaps(result, layer, FIG_DIR / f"attention_{lang}_layer{layer}.png")

    plot_head_statistics(results, FIG_DIR / "attention_head_statistics.png")
    plot_depth_trends(results, FIG_DIR / "attention_depth_trends.png")
    logger.info("done")


if __name__ == "__main__":
    main()
