"""
Choose the finetuning learning rate by a short probe, on validation data only.

    python -m common.finetune.lr_probe

Three candidate rates, one epoch each, on BOTH languages. The rate with the highest mean
validation accuracy across the two is used for both final runs.

Why probe at all
----------------
Pretraining could not afford a sweep; finetuning can, at about three minutes a run. And
the first measurement gave a specific reason not to trust the default: the gradient norm
sits at 6-7 against a clip threshold of 1.0, so every update is clipped and the nominal
learning rate overstates the step actually taken. What rate works best under that
clipping is an empirical question.

Why one rate for both models, chosen on both
--------------------------------------------
Phase 2 held optimisation identical between Model H and Model L so that differences
could be attributed to language and data. Picking a separate rate per language would
give up that control. Picking on Hindi alone and applying it to Sindhi would be worse
still — a quiet thumb on the scale for the higher-resource model. So the choice is made
on the mean over both.

Why validation and never test
-----------------------------
The test splits exist to measure the final models. Using them to choose a hyperparameter
would make the reported test accuracy an optimistic estimate, selected for. Only `val`
is read here.

Known limitation
----------------
The probe runs one epoch while the final runs use three, and the cosine schedule is
compressed to fit each. The ranking of rates is assumed to transfer from the short
horizon to the long one. That is standard practice for short probes but it is an
assumption, and is recorded as such.
"""

import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("lr_probe")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

LEARNING_RATES = [5e-5, 1e-4, 3e-4]
LANGUAGES = ["hindi", "sindhi"]
PROBE_EPOCHS = 1

# Rates within half a percentage point are treated as tied, and the LOWER rate wins. A
# smaller rate moves the weights less far from the pretrained solution, which is the
# safer choice when the task score cannot tell the difference.
TIE_MARGIN = 0.005


def probe_dir(language: str, lr: float) -> Path:
    return PROJECT_ROOT / language / "finetune" / "lr_probe" / f"lr_{lr:g}"


def run_one(language: str, lr: float) -> dict:
    """Run (or reuse) one probe and return its validation curve summary."""
    out = probe_dir(language, lr)
    done = out / "DONE"

    if not done.exists():
        # A probe with no completion marker was interrupted; three minutes is cheaper to
        # repeat than a partial run is to reason about.
        if out.exists():
            shutil.rmtree(out)
        logger.info("probe %s lr=%g", language, lr)
        subprocess.run(
            [sys.executable, "-m", "common.finetune.run",
             "--language", language, "--learning_rate", str(lr),
             "--epochs", str(PROBE_EPOCHS), "--out_dir", str(out)],
            check=True, cwd=PROJECT_ROOT)
        # The checkpoints are 280 MB each and a probe exists only to produce its curve.
        for ckpt in out.glob("*.pt"):
            ckpt.unlink()
        done.write_text("ok\n")
    else:
        logger.info("probe %s lr=%g already complete — reusing", language, lr)

    evals = [json.loads(line) for line in (out / "training_log.jsonl").open()
             if "val_accuracy" in line]
    return {
        "language": language, "learning_rate": lr,
        "best_val_accuracy": max(e["val_accuracy"] for e in evals),
        "final_val_accuracy": evals[-1]["val_accuracy"],
        "best_val_loss": min(e["val_loss"] for e in evals),
        "curve": [(e["step"], round(e["val_accuracy"], 4)) for e in evals],
    }


def main():
    runs = [run_one(lang, lr) for lr in LEARNING_RATES for lang in LANGUAGES]

    table = {}
    for lr in LEARNING_RATES:
        per_lang = {r["language"]: r["best_val_accuracy"] for r in runs
                    if r["learning_rate"] == lr}
        table[lr] = {"per_language": per_lang,
                     "mean": sum(per_lang.values()) / len(per_lang)}

    best_mean = max(v["mean"] for v in table.values())
    # Lowest rate within TIE_MARGIN of the best.
    chosen = min(lr for lr, v in table.items() if v["mean"] >= best_mean - TIE_MARGIN)

    logger.info("%-8s %10s %10s %10s", "lr", "hindi", "sindhi", "mean")
    for lr, v in table.items():
        mark = "  <- chosen" if lr == chosen else ""
        logger.info("%-8g %10.4f %10.4f %10.4f%s", lr, v["per_language"]["hindi"],
                    v["per_language"]["sindhi"], v["mean"], mark)

    out = PROJECT_ROOT / "report" / "finetune_lr_probe.json"
    out.write_text(json.dumps({
        "selection": "highest mean best-val-accuracy across both languages; "
                     f"ties within {TIE_MARGIN} resolved to the lower rate",
        "split_used": "val",
        "probe_epochs": PROBE_EPOCHS,
        "chosen_learning_rate": chosen,
        "summary": {str(lr): v for lr, v in table.items()},
        "runs": runs,
    }, indent=2) + "\n", encoding="utf-8")
    logger.info("wrote %s", out)
    print(f"CHOSEN_LR={chosen:g}")


if __name__ == "__main__":
    main()
