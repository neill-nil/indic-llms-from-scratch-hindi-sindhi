"""
Figures for the pretrained-vs-finetuned attention comparison.

Drawn from report/attention_finetune_results.json, written by attention_compare.py, so
figures can be restyled without re-running any model. Every figure carries a title, axis
labels and a legend or labelled colourbar — the brief warns that plots without them may
score zero.

    python -m common.finetune.plot_attention_compare
"""

import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common.eval.plot_attention import pick_font

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("plot_attention_compare")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS = PROJECT_ROOT / "report" / "attention_finetune_results.json"
FIG_DIR = PROJECT_ROOT / "report" / "figures"
DISPLAY = {"hindi": "Model H (Hindi)", "sindhi": "Model L (Sindhi)"}
CONDITION = {"pretrained": "pretrained", "finetuned": "finetuned"}


def fonts(language: str) -> list[str]:
    """
    Script font first, with a Latin fallback.

    A figure mixes English (titles, labels) with Devanagari or Perso-Arabic (token labels,
    the answer word). Noto Sans Devanagari has no Latin glyphs, so passing it alone
    rendered every English word in the title as empty boxes. Matplotlib falls back
    glyph-by-glyph through a family list, so both scripts render.
    """
    script = pick_font(language)
    return [script, "DejaVu Sans"] if script else ["DejaVu Sans"]


# Pretrained is dashed with hollow markers, finetuned solid with filled markers. The dash
# pattern alone was not enough: at legend size the short line segments rendered dashed and
# solid identically, so a reader could not tell which curve was which.
STYLE = {"pretrained": dict(linestyle="--", marker="o", markerfacecolor="white"),
         "finetuned": dict(linestyle="-", marker="o")}


def grid(entry: dict, condition: str, source: str, stat: str, split: str = "test_iid") -> np.ndarray:
    """(layers, heads) array of one statistic."""
    block = entry[condition]["corpus"] if source == "corpus" \
        else entry[condition]["reasoning"][split]["layers"]
    return np.array([block[str(l)][stat] for l in range(len(block))])


def changed_heads(entry: dict, layer: int, k: int = 4) -> list[int]:
    """
    The k heads in a layer whose answer-position attention to entity names changed most.

    Heads are selected by a stated rule rather than by eye, so the heatmaps cannot be
    chosen for looking dramatic; the selection rule is printed in the figure title.
    """
    delta = np.abs(grid(entry, "finetuned", "reasoning", "answer_entity_mass")[layer]
                   - grid(entry, "pretrained", "reasoning", "answer_entity_mass")[layer])
    return sorted(np.argsort(delta)[::-1][:k].tolist())


def plot_heatmaps(language: str, entry: dict, layer: int, out: Path) -> None:
    """Pretrained (top row) against finetuned (bottom row) on one reasoning prompt."""
    ex = entry["example"]
    tokens = [t.replace("▁", " ").strip() or "_" for t in ex["tokens"]]
    T = len(tokens)
    heads = changed_heads(entry, layer)
    font = fonts(language)
    gold_pos = set(ex["entity_positions"].get(ex["gold"], []))
    entity_pos = {p for ps in ex["entity_positions"].values() for p in ps}

    fig, axes = plt.subplots(2, len(heads), figsize=(4.3 * len(heads), 9.6),
                             constrained_layout=True)
    for row, condition in enumerate(("pretrained", "finetuned")):
        att = np.array(ex["attention"][condition][layer])       # (H, T, T)
        for col, h in enumerate(heads):
            ax = axes[row, col]
            m = att[h]
            # Clip at the 99th percentile of everything but the position-0 column, as in
            # Phase 2: the sink column would otherwise saturate the colour scale.
            body = m[:, 1:]
            vmax = float(np.percentile(body[body > 0], 99)) if (body > 0).any() else 1.0
            im = ax.imshow(m, cmap="viridis", vmin=0, vmax=max(vmax, 1e-6),
                           interpolation="nearest", aspect="equal")
            ax.set_title(f"{CONDITION[condition]} · layer {layer} · head {h}", fontsize=10)
            ax.set_xlabel("Key token (attended to)", fontsize=8)
            ax.set_ylabel("Query token (attending from)", fontsize=8)
            step = max(1, T // 24)
            ticks = list(range(0, T, step))
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)
            ax.set_xticklabels([tokens[i] for i in ticks], rotation=90, fontsize=5.5,
                               fontfamily=font)
            ax.set_yticklabels([tokens[i] for i in ticks], fontsize=5.5, fontfamily=font)
            for lab, i in zip(ax.get_xticklabels(), ticks):
                if i in gold_pos:
                    lab.set_color("red")
                elif i in entity_pos:
                    lab.set_color("darkorange")
            # The answer is produced from the final row; outline it.
            ax.add_patch(plt.Rectangle((-0.5, T - 1.5), T, 1, fill=False,
                                       edgecolor="red", linewidth=0.9))
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
            cbar.set_label("Attention weight (clipped at p99)", fontsize=7)

    fig.suptitle(
        f"{DISPLAY[language]} — attention before and after reasoning finetuning, layer "
        f"{layer} ({'early' if layer == 0 else 'late'})\n"
        f"Same prompt for both rows; correct answer: {ex['gold']}. Heads shown: the 4 in this "
        f"layer whose answer-position attention to entity names changed most.\n"
        f"Red outline = the answer position (final row). Tick labels: red = correct "
        f"entity, orange = other entities.", fontsize=11, fontfamily=font)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote %s", out.name)


def plot_head_changes(results: dict, out: Path) -> None:
    """Finetuned minus pretrained, per layer and head, for three statistics."""
    panels = [("corpus", "entropy_normalised", "Δ normalised entropy — corpus text"),
              ("corpus", "mean_distance", "Δ mean attention distance (tokens) — corpus text"),
              ("reasoning", "answer_entity_mass", "Δ answer-position mass on entity names — reasoning prompts")]
    fig, axes = plt.subplots(len(results), 3, figsize=(16, 4.4 * len(results)),
                             constrained_layout=True)
    axes = np.atleast_2d(axes)
    for r, (lang, entry) in enumerate(results.items()):
        for c, (source, stat, title) in enumerate(panels):
            d = grid(entry, "finetuned", source, stat) - grid(entry, "pretrained", source, stat)
            lim = float(np.abs(d).max()) or 1.0
            ax = axes[r, c]
            im = ax.imshow(d, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
            for (li, hi), v in np.ndenumerate(d):
                ax.text(hi, li, f"{v:+.2f}" if abs(v) < 10 else f"{v:+.0f}",
                        ha="center", va="center", fontsize=6.5)
            ax.set_title(f"{DISPLAY[lang]}: {title}", fontsize=10)
            ax.set_xlabel("Head")
            ax.set_ylabel("Layer")
            ax.set_xticks(range(d.shape[1]))
            ax.set_yticks(range(d.shape[0]))
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
            cbar.set_label("finetuned − pretrained  (red = increased)", fontsize=8)
    fig.suptitle("Where finetuning changed attention: every head, finetuned minus pretrained",
                 fontsize=13)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote %s", out.name)


def plot_depth(results: dict, out: Path) -> None:
    """Layer-wise means over heads, pretrained against finetuned, on corpus text."""
    stats = [("entropy_normalised", "Normalised entropy (0 = focused, 1 = uniform)"),
             ("mean_distance", "Mean attention distance (tokens)"),
             ("sink", "Attention mass on position 0 (sink)")]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), constrained_layout=True)
    colours = {"hindi": "tab:blue", "sindhi": "tab:orange"}
    for ax, (stat, ylabel) in zip(axes, stats):
        for lang, entry in results.items():
            for condition in ("pretrained", "finetuned"):
                g = grid(entry, condition, "corpus", stat).mean(axis=1)
                ax.plot(range(len(g)), g, color=colours[lang], **STYLE[condition],
                        label=f"{DISPLAY[lang]}, {condition}")
        ax.set_xlabel("Layer")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel.split(" (")[0] + " by layer", fontsize=11)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7.5, handlelength=3.5)
    fig.suptitle("Did finetuning change how the models read ordinary text? "
                 "Held-out corpus windows identical to Phase 2 (mean over heads)", fontsize=12)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote %s", out.name)


def plot_answer_position(results: dict, out: Path) -> None:
    """What the answer position attends to, by layer: entities overall, and gold vs distractor."""
    fig, axes = plt.subplots(len(results), 2, figsize=(14, 4.6 * len(results)),
                             constrained_layout=True)
    axes = np.atleast_2d(axes)
    split_label = {"test_iid": "names seen in finetuning", "test_entity": "held-out names"}
    for r, (lang, entry) in enumerate(results.items()):
        L = len(entry["pretrained"]["corpus"])
        for condition in ("pretrained", "finetuned"):
            for split, colour in (("test_iid", "tab:green"), ("test_entity", "tab:purple")):
                ent = grid(entry, condition, "reasoning", "answer_entity_mass", split).mean(axis=1)
                axes[r, 0].plot(range(L), ent, color=colour, **STYLE[condition],
                                label=f"{condition}, {split_label[split]}")
                # The discriminating signal lives in a few heads, so the strongest head
                # per layer is shown rather than a mean that would dilute it.
                gold = grid(entry, condition, "reasoning", "answer_gold_mass", split)
                dis = grid(entry, condition, "reasoning", "answer_distractor_mass", split)
                axes[r, 1].plot(range(L), (gold - dis).max(axis=1), color=colour,
                                **STYLE[condition], label=f"{condition}, {split_label[split]}")
        axes[r, 0].set_title(f"{DISPLAY[lang]}: answer-position attention on entity names "
                             f"(mean over heads)", fontsize=10)
        axes[r, 0].set_ylabel("Attention mass on entity-name tokens")
        axes[r, 1].set_title(f"{DISPLAY[lang]}: correct entity minus average distractor "
                             f"(strongest head per layer)", fontsize=10)
        axes[r, 1].set_ylabel("Mass on correct − mass on distractor")
        axes[r, 1].axhline(0, color="grey", linewidth=0.8)
        for ax in axes[r]:
            ax.set_xlabel("Layer")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=7.5, handlelength=3.5)
    fig.suptitle("At the moment of answering: does the model look at the entities, and at "
                 "the right one?", fontsize=13)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote %s", out.name)


def main():
    results = json.loads(RESULTS.read_text(encoding="utf-8"))
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for lang, entry in results.items():
        L = len(entry["pretrained"]["corpus"])
        for layer in (0, L - 1):
            plot_heatmaps(lang, entry, layer, FIG_DIR / f"attention_finetune_{lang}_layer{layer}.png")
    plot_head_changes(results, FIG_DIR / "attention_finetune_head_changes.png")
    plot_depth(results, FIG_DIR / "attention_finetune_depth.png")
    plot_answer_position(results, FIG_DIR / "attention_finetune_answer_position.png")


if __name__ == "__main__":
    main()
