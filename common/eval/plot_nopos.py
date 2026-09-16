"""
Figures for the no-positional-embeddings ablation.

    python -m common.eval.plot_nopos

Reads only committed results files. Produces:
  nopos_comparison.png    validation loss during training, bits per byte, generation quality
  nopos_attention.png     per-layer entropy, distance and previous-token mass
  attention_nopos_layer0.png / _layer5.png   heatmaps, via the Phase 2 plotting function

Every figure carries a title, axis labels and a legend or colourbar.
"""

import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common.eval.plot_attention import plot_heatmaps

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("plot_nopos")

ROOT = Path(__file__).resolve().parent.parent.parent
REPORT = ROOT / "report"
FIG = REPORT / "figures"

STD = {"label": "standard Model H (with positions)", "colour": "tab:blue",
       "style": dict(linestyle="-", marker="o", markersize=3)}
ABL = {"label": "no positional embeddings", "colour": "tab:red",
       "style": dict(linestyle="--", marker="s", markersize=3, markerfacecolor="white")}
TEMPS = [("greedy", "greedy"), ("temp_0.5", "T=0.5"), ("temp_1.0", "T=1.0"), ("temp_1.5", "T=1.5")]


def load(name):
    return json.loads((REPORT / name).read_text(encoding="utf-8"))


def hindi(rows):
    return [r for r in rows if r["language"] == "hindi"][0]


def val_curve(path):
    recs = [json.loads(l) for l in (ROOT / path).open() if "val_loss" in l]
    return [r["tokens"] for r in recs], [r["val_loss"] for r in recs]


def plot_comparison(out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8), constrained_layout=True)

    # (a) validation loss against tokens consumed
    for path, spec in (("hindi/train/run/training_log.jsonl", STD),
                       ("hindi/train/run_nopos/training_log.jsonl", ABL)):
        x, y = val_curve(path)
        axes[0].plot(np.array(x) / 1e6, y, color=spec["colour"], label=spec["label"], **spec["style"])
    axes[0].set_xlabel("Training tokens consumed (millions)")
    axes[0].set_ylabel("Validation loss (nats per token)")
    axes[0].set_title("Training: validation loss\n(hardware and precision also differ — see report)",
                      fontsize=10)

    # (b) bits per byte
    std = {(r["split"], r["context"]): r for r in load("intrinsic_results.json") if r["language"] == "hindi"}
    abl = {(r["split"], r["context"]): r for r in load("intrinsic_results_nopos.json") if r["language"] == "hindi"}
    keys = sorted(set(std) & set(abl))
    x = np.arange(len(keys))
    for i, (src, spec) in enumerate(((std, STD), (abl, ABL))):
        vals = [src[k]["bits_per_byte"] for k in keys]
        b = axes[1].bar(x + (i - 0.5) * 0.36, vals, 0.36, color=spec["colour"], label=spec["label"])
        axes[1].bar_label(b, labels=[f"{v:.3f}" for v in vals], fontsize=8, padding=1)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"{s} (ctx {c})" for s, c in keys])
    axes[1].set_xlabel("Held-out split")
    axes[1].set_ylabel("Bits per byte (lower is better)")
    axes[1].set_title("Language modelling", fontsize=10)

    # (c) generation quality across temperatures
    gs, ga = hindi(load("generation_results.json"))["metrics"], hindi(load("generation_results_nopos.json"))["metrics"]
    xs = np.arange(len(TEMPS))
    for src, spec in ((gs, STD), (ga, ABL)):
        axes[2].plot(xs, [src[k]["chrf++"] for k, _ in TEMPS], color=spec["colour"],
                     label=f"chrF++ — {spec['label']}", **spec["style"])
    ax2 = axes[2].twinx()
    for src, spec in ((gs, STD), (ga, ABL)):
        ax2.plot(xs, [src[k]["repetition_rate_4gram"] for k, _ in TEMPS], color=spec["colour"],
                 alpha=0.45, linestyle=":", marker="^", markersize=4,
                 label=f"repetition — {spec['label']}")
    axes[2].set_xticks(xs)
    axes[2].set_xticklabels([lab for _, lab in TEMPS])
    axes[2].set_xlabel("Decoding setting")
    axes[2].set_ylabel("chrF++ (higher is better)")
    ax2.set_ylabel("Repeated 4-grams (lower is better)")
    axes[2].set_title("Generation quality and repetition", fontsize=10)
    h1, l1 = axes[2].get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    axes[2].legend(h1 + h2, l1 + l2, fontsize=6.5, loc="upper left")

    for ax in axes[:2]:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[2].grid(alpha=0.3)
    fig.suptitle("Bonus ablation: what removing positional embeddings costs (Hindi, 500M tokens each)",
                 fontsize=12)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote %s", out.name)


def plot_attention(out: Path) -> None:
    std, abl = hindi(load("attention_results.json")), hindi(load("attention_results_nopos.json"))

    def stat(res, name):
        return np.array([res["layers"][str(l)][name] for l in range(res["n_layers"])])

    panels = [("entropy_normalised", "Normalised entropy (1 = uniform)"),
              ("mean_distance", "Mean attention distance (tokens)"),
              ("prev_token", "Attention mass on the previous token")]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), constrained_layout=True)
    for ax, (name, ylabel) in zip(axes, panels):
        for res, spec in ((std, STD), (abl, ABL)):
            g = stat(res, name)
            ax.plot(range(len(g)), g.mean(axis=1), color=spec["colour"], label=spec["label"],
                    **spec["style"])
            ax.fill_between(range(len(g)), g.min(axis=1), g.max(axis=1),
                            color=spec["colour"], alpha=0.12)
        ax.set_xlabel("Layer")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel.split(" (")[0] + " by layer", fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Attention with and without positional embeddings — line is the mean over the 8 "
                 "heads, shaded band spans them", fontsize=12)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote %s", out.name)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    plot_comparison(FIG / "nopos_comparison.png")
    plot_attention(FIG / "nopos_attention.png")
    # Heatmaps through the Phase 2 function, so they are drawn exactly like Phase 2's.
    abl = hindi(load("attention_results_nopos.json"))
    for layer in (0, abl["n_layers"] - 1):
        plot_heatmaps(abl, layer, FIG / f"attention_nopos_layer{layer}.png")


if __name__ == "__main__":
    main()
