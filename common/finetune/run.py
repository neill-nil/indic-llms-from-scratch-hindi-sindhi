"""
Reasoning finetuning entry point.

    python -m common.finetune.run --language hindi
    python -m common.finetune.run --language hindi --resume

Resume is idempotent, exactly as in pretraining: re-running after an interruption picks
up from latest.pt and stops at the same step count.
"""

import argparse
import json
import logging
import signal
import sys
from dataclasses import asdict
from pathlib import Path

import torch

from common.finetune.dataset import load_reasoning_split
from common.finetune.trainer import FinetuneConfig, FinetuneTrainer
from common.model.config import GPTConfig
from common.model.gpt import GPT

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--language", required=True, choices=["hindi", "sindhi"])
    p.add_argument("--out_dir", default=None, help="default: <language>/finetune/run")
    p.add_argument("--pretrained", default=None,
                   help="default: <language>/train/run/best.pt")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--epochs", type=int, default=FinetuneConfig.epochs)
    p.add_argument("--learning_rate", type=float, default=FinetuneConfig.learning_rate)
    p.add_argument("--micro_batch", type=int, default=FinetuneConfig.micro_batch)
    p.add_argument("--grad_accum", type=int, default=FinetuneConfig.grad_accum)
    p.add_argument("--seed", type=int, default=FinetuneConfig.seed)
    p.add_argument("--loss_on", default="answer", choices=["answer", "full"],
                   help="supervise the answer only (default) or every token")
    p.add_argument("--limit", type=int, default=None,
                   help="use only the first N training examples (smoke tests)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else \
        PROJECT_ROOT / args.language / "finetune" / "run"
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout),
                  logging.FileHandler(out_dir / "finetune.log", mode="a",
                                      encoding="utf-8")])
    log = logging.getLogger("finetune")

    # Before the model is built: GPT.__init__ draws from the global RNG.
    torch.manual_seed(args.seed)
    if args.device.startswith("cuda"):
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    model_cfg = GPTConfig.from_json(
        PROJECT_ROOT / args.language / "configs" / "model_config.json")
    model = GPT(model_cfg)

    train_data = load_reasoning_split(args.language, "train", limit=args.limit,
                                      loss_on=args.loss_on)
    # Validation is ALWAYS scored on the answer only, whatever the training loss. Its
    # exact-answer accuracy counts an example correct when every supervised token is right;
    # supervising the prompt there would demand the whole question be predicted, and the
    # two runs' validation curves would stop being comparable.
    val_data = load_reasoning_split(args.language, "val")

    # The cosine floor is tied to the peak (10%, as in pretraining) rather than fixed.
    # A fixed floor would give each learning rate a different schedule SHAPE — 20% of peak
    # at 5e-5 but 3% at 3e-4 — so a learning-rate comparison would also be quietly
    # comparing how far each run anneals.
    cfg = FinetuneConfig(epochs=args.epochs, learning_rate=args.learning_rate,
                         loss_on=args.loss_on,
                         min_lr=args.learning_rate / 10,
                         micro_batch=args.micro_batch, grad_accum=args.grad_accum,
                         seed=args.seed)
    trainer = FinetuneTrainer(model, train_data, val_data, cfg, out_dir, args.device)

    latest = out_dir / "latest.pt"
    if args.resume:
        if not latest.exists():
            log.error("--resume given but %s does not exist", latest)
            raise SystemExit(1)
        trainer.load_checkpoint(latest)
    elif latest.exists():
        log.error("%s exists. Pass --resume, or choose another --out_dir.", latest)
        raise SystemExit(1)
    else:
        pretrained = Path(args.pretrained) if args.pretrained else \
            PROJECT_ROOT / args.language / "train" / "run" / "best.pt"
        trainer.init_from_pretrained(pretrained)

    model_cfg.to_json(out_dir / "model_config.json")
    # The optimisation settings too, as a committed file. Until this was added they existed
    # only inside the checkpoints — which are not in git — so the hyperparameters of a run
    # could not be read from the repository at all. Written after the trainer is built,
    # because the trainer derives the real token budget from the epoch count.
    (out_dir / "train_config.json").write_text(
        json.dumps(asdict(trainer.cfg), indent=2) + "\n", encoding="utf-8")

    def _save_and_exit(signum, _frame):
        log.warning("received signal %s", signal.Signals(signum).name)
        raise KeyboardInterrupt
    for sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, _save_and_exit)

    try:
        trainer.train()
    except KeyboardInterrupt:
        log.warning("interrupted at step %d — saving", trainer.step)
        trainer.save_checkpoint("latest")
        raise SystemExit(130)


if __name__ == "__main__":
    main()
