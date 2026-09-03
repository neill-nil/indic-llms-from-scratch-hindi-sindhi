"""
Pretraining entry point.

    python -m common.train.train --language hindi
    python -m common.train.train --language hindi --resume        # continue a run
    python -m common.train.train --language sindhi --micro_batch 16 --grad_accum 3

Resume is idempotent: pointing it at an existing run picks up from latest.pt and stops at
the same token budget, so re-running after an interruption is always safe.
"""

import argparse
import logging
import signal
import sys
from pathlib import Path

import torch

from common.data.loader import load_split
from common.model.config import GPTConfig
from common.model.gpt import GPT
from common.train.trainer import TrainConfig, Trainer, select_precision

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def setup_logging(out_dir: Path) -> None:
    """Log to stdout and to a file, so a detached run leaves a readable record."""
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout),
                  logging.FileHandler(out_dir / "train.log", mode="a", encoding="utf-8")],
    )


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--language", required=True, choices=["hindi", "sindhi"])
    p.add_argument("--out_dir", type=str, default=None,
                   help="default: <language>/train/run")
    p.add_argument("--data_dir", type=str, default=None,
                   help="directory holding train.bin / val.bin; default is the repo layout")
    p.add_argument("--resume", action="store_true", help="continue from latest.pt")

    p.add_argument("--max_tokens", type=int, default=TrainConfig.max_tokens)
    p.add_argument("--micro_batch", type=int, default=TrainConfig.micro_batch)
    p.add_argument("--grad_accum", type=int, default=TrainConfig.grad_accum)
    p.add_argument("--learning_rate", type=float, default=TrainConfig.learning_rate)
    p.add_argument("--warmup_steps", type=int, default=TrainConfig.warmup_steps)
    p.add_argument("--eval_every", type=int, default=TrainConfig.eval_every)
    p.add_argument("--checkpoint_every", type=int, default=TrainConfig.checkpoint_every)
    p.add_argument("--seed", type=int, default=TrainConfig.seed)
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else PROJECT_ROOT / args.language / "train" / "run"
    setup_logging(out_dir)
    log = logging.getLogger("train")

    # ── Seed ────────────────────────────────────────────────────────────────────
    # Must happen before the model is constructed: GPT.__init__ draws every weight from
    # the global RNG, so seeding afterwards leaves initialisation uncontrolled and two
    # runs with identical arguments start from different weights.
    torch.manual_seed(args.seed)
    if args.device.startswith("cuda"):
        torch.cuda.manual_seed_all(args.seed)
        # TF32 matmuls: a large speedup on Ampere for a precision loss well below the
        # noise floor of bf16 training.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # ── Model ───────────────────────────────────────────────────────────────────
    model_cfg = GPTConfig.from_json(PROJECT_ROOT / args.language / "configs" / "model_config.json")
    model = GPT(model_cfg)
    log.info("model config: %s", model_cfg)
    log.info("\n%s", model.parameter_summary())

    # ── Data ────────────────────────────────────────────────────────────────────
    if args.data_dir:
        from common.data.loader import TokenDataset
        data_dir = Path(args.data_dir)
        train_data = TokenDataset(data_dir / "train.bin", model_cfg.max_seq_len)
        val_data = TokenDataset(data_dir / "val.bin", model_cfg.max_seq_len)
    else:
        train_data = load_split(args.language, "train", model_cfg.max_seq_len)
        val_data = load_split(args.language, "val", model_cfg.max_seq_len)
    log.info("data: train %s tokens, val %s tokens",
             f"{train_data.n_tokens:,}", f"{val_data.n_tokens:,}")

    if args.max_tokens > train_data.n_tokens:
        log.warning("token budget %s exceeds the training split (%s tokens); "
                    "windows will be revisited",
                    f"{args.max_tokens:,}", f"{train_data.n_tokens:,}")

    # ── Trainer ─────────────────────────────────────────────────────────────────
    train_cfg = TrainConfig(
        max_tokens=args.max_tokens,
        micro_batch=args.micro_batch,
        grad_accum=args.grad_accum,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        eval_every=args.eval_every,
        checkpoint_every=args.checkpoint_every,
        seed=args.seed,
    )

    trainer = Trainer(model, train_data, val_data, train_cfg, out_dir, args.device)

    dtype, scaler = select_precision(args.device)
    log.info("device %s | autocast %s%s", args.device, dtype,
             " + GradScaler" if scaler else "")

    latest = out_dir / "latest.pt"
    if args.resume:
        if not latest.exists():
            log.error("--resume given but %s does not exist", latest)
            raise SystemExit(1)
        trainer.load_checkpoint(latest)
    elif latest.exists():
        log.error("%s already exists. Pass --resume to continue it, or choose a different "
                  "--out_dir to start fresh. Refusing to overwrite an existing run.", latest)
        raise SystemExit(1)

    # Write the config next to the checkpoints, as the brief requires.
    model_cfg.to_json(out_dir / "model_config.json")

    # Catch the signals that end a long unattended run so it saves before dying.
    #   SIGINT  — Ctrl-C
    #   SIGTERM — `kill`, and what systemd sends to user processes at shutdown or
    #             when a session ends
    #   SIGHUP  — the controlling terminal went away (closing an SSH session or a
    #             terminal window that was not started with nohup)
    # Each is re-raised as KeyboardInterrupt so a single handler below covers all of
    # them. Note this cannot protect against SIGKILL or against the GPU context being
    # invalidated by a suspend: only the periodic checkpoint covers those, which is why
    # --checkpoint_every should be small for laptop runs.
    def _save_and_exit(signum, _frame):
        log.warning("received signal %s", signal.Signals(signum).name)
        raise KeyboardInterrupt
    for sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, _save_and_exit)

    try:
        trainer.train()
    except KeyboardInterrupt:
        log.warning("interrupted at step %d — saving checkpoint before exit", trainer.step)
        trainer.save_checkpoint("latest")
        log.warning("saved %s", out_dir / "latest.pt")
        log.warning("resume with:  bash scripts/train_local.sh %s", args.language)
        raise SystemExit(130)


if __name__ == "__main__":
    main()
