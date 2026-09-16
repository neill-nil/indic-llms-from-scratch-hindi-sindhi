"""
One compact figure covering the brief's attention requirement, for a page-limited report.

    python -m common.finetune.plot_attention_compact

The brief asks for pretrained-vs-finetuned heatmaps at an early and a late layer for both
models. As four separate figures that is most of a page each. This packs the same content
into a single 2x4 grid: one row per model, and within a row, early-pretrained,
early-finetuned, late-pretrained, late-finetuned.

One head is shown per panel — the head in that layer whose answer-position attention to
entity names changed most between the two checkpoints, which is a stated rule rather than a
hand-picked choice. The full four-figure version (all four heads per layer) remains in
attention_finetune_{language}_layer{0,5}.png.
"""

import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common.finetune.plot_attention_compare import RESULTS, changed_heads, fonts, DISPLAY

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("plot_attention_compact")

FIG = Path(__file__).resolve().parent.parent.parent / "report" / "figures"


def main():
    results = json.loads(RESULTS.read_text(encoding="utf-8"))
    languages = list(results)
    fig, axes = plt.subplots(len(languages), 4, figsize=(17, 4.6 * len(languages)),
                             constrained_layout=True)
    axes = np.atleast_2d(axes)

    for row, lang in enumerate(languages):
        entry = results[lang]
        ex = entry["example"]
        tokens = [t.replace("▁", " ").strip() or "_" for t in ex["tokens"]]
        T = len(tokens)
        font = fonts(lang)
        gold = set(ex["entity_positions"].get(ex["gold"], []))
        ents = {p for ps in ex["entity_positions"].values() for p in ps}
        n_layers = len(entry["pretrained"]["corpus"])

        col = 0
        for layer in (0, n_layers - 1):
            head = changed_heads(entry, layer, k=1)[0]
            for condition in ("pretrained", "finetuned"):
                ax = axes[row, col]
                m = np.array(ex["attention"][condition][layer])[head]
                body = m[:, 1:]
                vmax = float(np.percentile(body[body > 0], 99)) if (body > 0).any() else 1.0
                im = ax.imshow(m, cmap="viridis", vmin=0, vmax=max(vmax, 1e-6),
                               interpolation="nearest", aspect="equal")
                ax.set_title(f"{'early' if layer == 0 else 'late'} — layer {layer}, head {head}\n"
                             f"{condition}", fontsize=9)
                ax.set_xlabel("Key token (attended to)", fontsize=7)
                if col == 0:
                    ax.set_ylabel(f"{DISPLAY[lang]}\nQuery token (attending from)", fontsize=8)
                step = max(1, T // 14)
                ticks = list(range(0, T, step))
                ax.set_xticks(ticks); ax.set_yticks(ticks)
                ax.set_xticklabels([tokens[i] for i in ticks], rotation=90, fontsize=5,
                                   fontfamily=font)
                ax.set_yticklabels([tokens[i] for i in ticks], fontsize=5, fontfamily=font)
                for lab, i in zip(ax.get_xticklabels(), ticks):
                    if i in gold:
                        lab.set_color("red")
                    elif i in ents:
                        lab.set_color("darkorange")
                ax.add_patch(plt.Rectangle((-0.5, T - 1.5), T, 1, fill=False,
                                           edgecolor="red", linewidth=0.8))
                cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
                cbar.set_label("Attention weight (clipped at p99)", fontsize=6)
                col += 1

    fig.suptitle(
        "Attention before and after reasoning finetuning — early and late layer, both models.\n"
        "One reasoning prompt; the head shown per layer is the one whose answer-position "
        "attention to entity names changed most. Red outline = the answer position; "
        "red tick = the correct entity, orange = other entities.", fontsize=11)
    out = FIG / "attention_finetune_compact.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote %s", out.name)


if __name__ == "__main__":
    main()
