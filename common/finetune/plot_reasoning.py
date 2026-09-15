"""
Phase 3 figures: finetuning curves, the learning-rate probe, reasoning accuracy against its
chance floor, what the models get right, and what finetuning cost the language model.

Drawn only from committed results files, never from a model, so every figure can be
regenerated in seconds and checked against the numbers it shows.

    python -m common.finetune.plot_reasoning

Every figure has a title, axis labels and a legend — the brief warns that plots missing
them may receive zero marks for that component.
"""

import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("plot_reasoning")

ROOT = Path(__file__).resolve().parent.parent.parent
REPORT = ROOT / "report"
FIG = REPORT / "figures"

LANGS = ["hindi", "sindhi"]
DISPLAY = {"hindi": "Model H (Hindi)", "sindhi": "Model L (Sindhi)"}
COLOUR = {"hindi": "tab:blue", "sindhi": "tab:orange"}
SPLITS = ["test_iid", "test_entity", "test_template", "test_hop"]
SPLIT_LABEL = {"test_iid": "in-distribution", "test_entity": "held-out names",
               "test_template": "held-out phrasings", "test_hop": "longer problems"}


def load(name: str):
    return json.loads((REPORT / name).read_text(encoding="utf-8"))


def by_language(results: list) -> dict:
    return {r["language"]: r for r in results}


def log_records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8")]


def smooth(values: list[float], k: int = 9) -> np.ndarray:
    """Centred moving average; the per-batch training loss is too noisy to read raw."""
    v = np.asarray(values, dtype=float)
    if len(v) < k:
        return v
    pad = k // 2
    padded = np.pad(v, pad, mode="edge")
    return np.convolve(padded, np.ones(k) / k, mode="valid")


def plot_finetune_curves(out: Path) -> None:
    """Training loss, validation loss and validation accuracy for every finetuning run."""
    analysis = load("reasoning_analysis.json")
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8), constrained_layout=True)
    for lang in LANGS:
        for run, style, label in (("run", "-", "loss on answer (main model)"),
                                  ("run_fullseq", "--", "loss on full sequence (ablation)")):
            recs = log_records(ROOT / lang / "finetune" / run / "training_log.jsonl")
            train = [r for r in recs if "train_loss" in r]
            val = [r for r in recs if "val_accuracy" in r]
            name = f"{DISPLAY[lang]}, {label}"
            axes[0].plot([r["step"] for r in train], smooth([r["train_loss"] for r in train]),
                         style, color=COLOUR[lang], label=name)
            axes[1].plot([r["step"] for r in val], [r["val_loss"] for r in val], style,
                         marker="o", markersize=3, color=COLOUR[lang], label=name)
            axes[2].plot([r["step"] for r in val], [r["val_accuracy"] for r in val], style,
                         marker="o", markersize=3, color=COLOUR[lang], label=name)
        chance = analysis[f"{lang}/finetuned_0shot"]["chance_by_split"]["test_iid"]["random_name"]
        axes[2].axhline(chance, color=COLOUR[lang], linestyle=":", linewidth=1.2,
                        label=f"{DISPLAY[lang]}: chance (random name from the question)")

    axes[0].set_title("Training loss (smoothed)\nfull-sequence runs include prompt tokens, so "
                      "their scale differs", fontsize=10)
    axes[0].set_ylabel("Cross-entropy (nats)")
    axes[1].set_title("Validation loss on answer tokens\n(same quantity for every run)", fontsize=10)
    axes[1].set_ylabel("Cross-entropy (nats)")
    axes[2].set_title("Validation exact-answer accuracy (teacher forced)", fontsize=10)
    axes[2].set_ylabel("Fraction of answers fully correct")
    for i, ax in enumerate(axes):
        ax.set_xlabel("Optimizer step (3 epochs = 3,750 steps)")
        ax.grid(alpha=0.3)
        # Accuracy rises towards the top-left, so its legend goes where the curves are not.
        ax.legend(fontsize=6.5, handlelength=3, loc="lower right" if i == 2 else "best")
    fig.suptitle("Reasoning finetuning curves — both models, lr 1e-4, identical settings",
                 fontsize=12)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote %s", out.name)


def plot_lr_probe(out: Path) -> None:
    probe = load("finetune_lr_probe.json")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), constrained_layout=True, sharey=True)
    styles = {5e-5: ":", 1e-4: "-", 3e-4: "--"}
    for ax, lang in zip(axes, LANGS):
        for run in (r for r in probe["runs"] if r["language"] == lang):
            steps, acc = zip(*run["curve"])
            chosen = " (chosen)" if run["learning_rate"] == probe["chosen_learning_rate"] else ""
            ax.plot(steps, acc, styles[run["learning_rate"]], marker="o", markersize=3,
                    color=COLOUR[lang], label=f"lr {run['learning_rate']:g}{chosen}")
        ax.set_title(f"{DISPLAY[lang]}", fontsize=11)
        ax.set_xlabel("Optimizer step (1 epoch = 1,250 steps)")
        ax.set_ylabel("Validation exact-answer accuracy")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, handlelength=3)
    fig.suptitle("Learning-rate probe: one epoch per rate, validation split only; one rate "
                 "chosen for both models", fontsize=12)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote %s", out.name)


def plot_accuracy(out: Path) -> None:
    """Test accuracy for every condition, with the chance floor it must be read against."""
    greedy = {c: by_language(load(f"reasoning_{c}.json")) for c in
              ("pretrained_0shot", "pretrained_4shot", "finetuned_0shot")}
    forced = by_language(load("reasoning_forced_choice_finetuned.json"))
    analysis = load("reasoning_analysis.json")

    bars = [("pretrained, zero-shot", lambda l, s: greedy["pretrained_0shot"][l]["splits"][s]["accuracy"]),
            ("pretrained, 4-shot", lambda l, s: greedy["pretrained_4shot"][l]["splits"][s]["accuracy"]),
            ("finetuned, greedy generation", lambda l, s: greedy["finetuned_0shot"][l]["splits"][s]["accuracy"]),
            ("finetuned, forced choice among names", lambda l, s: forced[l]["splits"][s]["names_only"]["accuracy"])]
    colours = ["#bbbbbb", "#888888", "tab:green", "tab:purple"]

    fig, axes = plt.subplots(1, 2, figsize=(15, 5), constrained_layout=True, sharey=True)
    x = np.arange(len(SPLITS))
    w = 0.2
    for ax, lang in zip(axes, LANGS):
        for i, ((label, get), c) in enumerate(zip(bars, colours)):
            vals = [get(lang, s) for s in SPLITS]
            b = ax.bar(x + (i - 1.5) * w, vals, w, color=c, label=label)
            ax.bar_label(b, labels=[f"{v:.2f}" for v in vals], fontsize=6.5, padding=1)
        for j, s in enumerate(SPLITS):
            ch = analysis[f"{lang}/finetuned_0shot"]["chance_by_split"][s]["random_name"]
            ax.hlines(ch, j - 2 * w, j + 2 * w, color="red", linestyle="--", linewidth=1.3,
                      label="chance: random name from the question" if j == 0 else None)
        ax.set_xticks(x)
        ax.set_xticklabels([SPLIT_LABEL[s] for s in SPLITS])
        ax.set_title(DISPLAY[lang], fontsize=11)
        ax.set_xlabel("Test split")
        ax.set_ylabel("Exact-match accuracy")
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.3)
    # One shared legend below the panels. Inside the axes it covered the tallest bars and
    # their value labels.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=5, fontsize=8.5)
    fig.suptitle("Reasoning accuracy by test split. Pretrained models score exactly 0; the fair "
                 "floor for a finetuned model is chance (red).", fontsize=12)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote %s", out.name)


def task_chance(split: str = "test_iid") -> list[float]:
    """
    Chance per task type: a uniform guess among every answer the question allows.

    Computed from the test data rather than written in, because the candidate set differs by
    task — the entity names plus "equal" for a numeric comparison, the names alone for a
    chain, and "more"/"less" for a relation question — and so does the number of entities.
    An earlier version hard-coded approximate floors, one of them wrong by eight points.
    Both languages share task structure, so Hindi's test split is used for both.
    """
    from common.finetune.analyse_reasoning import load_lexicon, load_split, entities_in_order
    lex = load_lexicon("hindi")
    groups: dict[str, list[float]] = {}
    for ex in load_split("hindi", split):
        n = len(entities_in_order(ex, lex))
        if ex["family"] == "value_compare":
            key = "numeric, tie" if ex["is_tie"] else "numeric, no tie"
            p = 1 / (n + 1)
        elif ex["question_type"] == "relation":
            key, p = "chain, relation", 0.5
        else:
            key, p = f"chain, {ex['question_type']}", 1 / n
        groups.setdefault(key, []).append(p)
    order = ["numeric, no tie", "numeric, tie", "chain, argmax", "chain, argmin", "chain, relation"]
    return [float(np.mean(groups[k])) for k in order]


def plot_what_is_learned(out: Path) -> None:
    """Controlled in-distribution breakdowns: margin effect, and accuracy by task type."""
    analysis = load("reasoning_analysis.json")
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.8), constrained_layout=True)
    tasks = ["numeric, no tie", "numeric, tie", "chain, argmax", "chain, argmin", "chain, relation"]
    chance = task_chance()
    x = np.arange(len(tasks))
    for i, lang in enumerate(LANGS):
        c = analysis[f"{lang}/finetuned_0shot"]["controlled_in_distribution"]
        bins = list(c["accuracy_by_margin"])
        axes[0].plot(range(len(bins)), [c["accuracy_by_margin"][b]["accuracy"] for b in bins],
                     marker="o", color=COLOUR[lang], label=DISPLAY[lang])
        vals = [c["accuracy_by_task"][t]["accuracy"] for t in tasks]
        bb = axes[1].bar(x + (i - 0.5) * 0.38, vals, 0.38, color=COLOUR[lang], label=DISPLAY[lang])
        axes[1].bar_label(bb, labels=[f"{v:.2f}" for v in vals], fontsize=7, padding=1)
    axes[0].set_xticks(range(len(bins)))
    axes[0].set_xticklabels(bins)
    axes[0].set_xlabel("Gap between best and second-best value (share of the attribute's range)")
    axes[0].set_ylabel("Exact-match accuracy")
    axes[0].set_title("Harder when the numbers are close — evidence of genuine comparison",
                      fontsize=10)
    axes[0].set_ylim(0, 1.05)
    for j, ch in enumerate(chance):
        axes[1].hlines(ch, j - 0.4, j + 0.4, color="red", linestyle="--", linewidth=1.2,
                       label="chance: uniform guess among possible answers" if j == 0 else None)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(tasks, fontsize=9)
    axes[1].set_xlabel("Task type")
    axes[1].set_ylabel("Exact-match accuracy")
    axes[1].set_title("The H–L gap sits in chains; neither model learned the relation question",
                      fontsize=10)
    axes[1].set_ylim(0, 1.05)
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("What the finetuned models learned — in-distribution test split only, so "
                 "unseen-name failures do not dilute it", fontsize=12)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote %s", out.name)


def plot_lm_regression(out: Path) -> None:
    """Bits per byte on ordinary held-out text, before and after finetuning."""
    def bpb(name, lang, ctx):
        rows = [r for r in load(name) if r["language"] == lang and r["split"] == "val"
                and r["context"] == ctx]
        return rows[0]["bits_per_byte"]

    cases = [("hindi", 512), ("sindhi", 1024)]
    conditions = [("pretrained", "intrinsic_results.json", "#888888"),
                  ("finetuned, loss on answer", "intrinsic_results_finetuned.json", "tab:red"),
                  ("finetuned, loss on full sequence", "intrinsic_results_finetuned_fullseq.json", "darkred")]
    fig, ax = plt.subplots(figsize=(10, 4.8), constrained_layout=True)
    x = np.arange(len(cases))
    for i, (label, name, colour) in enumerate(conditions):
        vals = [bpb(name, lang, ctx) for lang, ctx in cases]
        b = ax.bar(x + (i - 1) * 0.27, vals, 0.27, color=colour, label=label)
        ax.bar_label(b, labels=[f"{v:.3f}" for v in vals], fontsize=8, padding=1)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{DISPLAY[l]}, validation, context {c}" for l, c in cases])
    ax.set_xlabel("Model and held-out split")
    ax.set_ylabel("Bits per byte (lower is better)")
    ax.set_title("Finetuning at least doubled bits per byte on ordinary text, with either loss",
                 fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote %s", out.name)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    plot_finetune_curves(FIG / "finetune_curves.png")
    plot_lr_probe(FIG / "finetune_lr_probe.png")
    plot_accuracy(FIG / "reasoning_accuracy.png")
    plot_what_is_learned(FIG / "reasoning_what_is_learned.png")
    plot_lm_regression(FIG / "finetune_lm_regression.png")


if __name__ == "__main__":
    main()
